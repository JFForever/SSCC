import json

import os

import math

import cv2

import torch

import torch.nn as nn

from torch.nn import functional as F

from torch.utils.checkpoint import checkpoint as grad_checkpoint



from clip import clip

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.ot import (                             

    build_caption_mask,

    build_similarity_source_mass,

    compute_ot_transport,

    cosine_cost_matrix,

    transport_barycentric_projection,

)



                                               

_tokenizer = _Tokenizer()

                                                                 

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0



def load_clip_to_cpu(cfg):  

       

                    

                                                               

                                                 

                                             

       

    backbone_name = cfg.MODEL.BACKBONE.NAME

    url = clip._MODELS[backbone_name]

    local_model_path = os.environ.get("RCNN_CLIP_MODEL_PATH", "/mnt/a/cs_ai_wys/GJF/clip/ViT-B-16.pt")

    if local_model_path and os.path.exists(local_model_path):

        model_path = local_model_path

    else:

        model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))



    try:

                                                   

        model = torch.jit.load(model_path, map_location="cpu").eval()

        state_dict = None



    except RuntimeError:

        state_dict = torch.load(model_path, map_location="cpu")

    

                                               

    design_details = {"trainer": 'IVLP',

                      "vision_depth": 0, "vision_ctx": 0,

                      "language_depth": 0, "language_ctx": 0,                               

                      "last_enabled": bool(getattr(cfg.LAST, "ENABLED", False)),                               

                      "last_mode": str(getattr(cfg.LAST, "MODE", "replace")),

                      "last_topk": int(getattr(cfg.LAST, "TOPK", 0)),

                      "last_topk_ratio": float(getattr(cfg.LAST, "TOPK_RATIO", 0.5)),

                      "last_fusion_alpha": float(getattr(cfg.LAST, "FUSION_ALPHA", 0.1)),

                      "last_sigma": float(getattr(cfg.LAST, "SIGMA", 0.0)),

                      "last_eps": float(getattr(cfg.LAST, "EPS", 1e-6)),

                      }                                 

                          

    model = clip.build_model(state_dict or model.state_dict(), design_details)                       



    if hasattr(model.visual, "set_last_pool_config"):                                                                

        model.visual.set_last_pool_config(

            enabled=bool(getattr(cfg.LAST, "ENABLED", False)),

            mode=str(getattr(cfg.LAST, "MODE", "replace")),

            topk=int(getattr(cfg.LAST, "TOPK", 0)),

            topk_ratio=float(getattr(cfg.LAST, "TOPK_RATIO", 0.5)),

            fusion_alpha=float(getattr(cfg.LAST, "FUSION_ALPHA", 0.1)),

            sigma=float(getattr(cfg.LAST, "SIGMA", 0.0)),

            eps=float(getattr(cfg.LAST, "EPS", 1e-6)),

        )



    return model



class TextEncoder(nn.Module):

       

                 



              

                                                           

                                         



              

                                     

                                                          

       

    def __init__(self, clip_model):                                          

        super().__init__()

        self.transformer = clip_model.transformer

        self.positional_embedding = clip_model.positional_embedding

        self.ln_final = clip_model.ln_final

        self.text_projection = clip_model.text_projection

        self.dtype = clip_model.dtype

        self.token_embedding = clip_model.token_embedding



                                              

    def forward(self, prompts, tokenized_prompts, if_embedding=True, return_sequence=False):

           

           

                

                                                                                       

                                                                               

                          

                                             

                                                                                   

                     

                                   

                                                       

                        

                                            

                                                                           

                   

        

        if not if_embedding:                                                        

            tokenized_prompts = prompts

            prompts = self.token_embedding(prompts).type(self.dtype)



        x = prompts + self.positional_embedding.type(self.dtype)                            

        x = x.permute(1, 0, 2)                                            

        if x.requires_grad:       

            x = grad_checkpoint(self.transformer, x, use_reentrant=False)

        else:

            x = self.transformer(x)                            

        x = x.permute(1, 0, 2)                                                                  

        x = self.ln_final(x).type(self.dtype)



                                          

        if tokenized_prompts.device != x.device:

            tokenized_prompts = tokenized_prompts.to(x.device)



                                           

                                                     

        eot_indices = tokenized_prompts.argmax(dim=-1)                 

                                                                             

        if eot_indices.shape[0] == 1 and x.shape[0] != 1:

            eot_indices = eot_indices.expand(x.shape[0])

        elif eot_indices.shape[0] != x.shape[0]:

            raise ValueError(

                "tokenized_prompts batch dimension must match encoded prompts or be 1, "

                f"got {eot_indices.shape[0]} and {x.shape[0]}"

            )



                                                               

        pooled = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection

        if not return_sequence:                                                     

            return pooled

                                             

        seq = x @ self.text_projection

        return pooled, seq



class ClsnameLearner(nn.Module):                                             

       

                        



         

                                   



                 

                                                   



              

                 

                            



                                                                                

       

    def __init__(self, cfg, cls_num, clip_model):

        super().__init__()

        self.dtype = clip_model.dtype

        n_clsn = cfg.TRAINER.TEXT.N_CLSN                                               

        ctx_init = cfg.TRAINER.TEXT.CTX_INIT                              

        dtype = clip_model.dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]                                                



        ctx_init = ctx_init.replace("_", " ")                               

        n_ctx = len(ctx_init.split(" "))                  



        

        clsn_vectors = torch.empty(cls_num, n_clsn, ctx_dim, dtype=dtype)                                                                                                                                                           

        nn.init.normal_(clsn_vectors, std=0.02)                                

        clsn_sign = " ".join(["X"] * n_clsn)                                     

        self.clsn = nn.Parameter(clsn_vectors)                                              



                                                 

        prompt = ctx_init + " " + clsn_sign + "."

        tokenized_prompt = clip.tokenize(prompt)                                                         

                                             

        with torch.no_grad():                                

            embedding = clip_model.token_embedding(tokenized_prompt).type(dtype)                                                           



        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])                                                               

        self.register_buffer("token_suffix", embedding[:, n_ctx + n_clsn + 1:, :])                                      

        self.cls_num = cls_num           

        self.n_clsn = n_clsn

        

        self.learnable_start = n_ctx + 1                                     

        self.learnable_end = self.learnable_start + n_clsn                                  

        

        self.tokenized_prompt = tokenized_prompt                                            



    def get_class_tokens(self):

                            

        clsn = self.clsn

                                               

        if clsn.dim() == 2:

            clsn = clsn.unsqueeze(0).expand(self.cls_num, -1, -1)

        return clsn



    def forward(self, class_indices=None):

           

                                 



           

                                               

                   

        clsn = self.get_class_tokens()

        prefix = self.token_prefix

        suffix = self.token_suffix



                                            

        if class_indices is not None:

            class_indices = class_indices.to(device=clsn.device, dtype=torch.long)

            clsn = clsn.index_select(0, class_indices)

            if prefix.shape[0] == 1:

                prefix = prefix.expand(class_indices.numel(), -1, -1)

                suffix = suffix.expand(class_indices.numel(), -1, -1)

            else:

                prefix = prefix.index_select(0, class_indices)

                suffix = suffix.index_select(0, class_indices)

        elif prefix.shape[0] == 1:

            prefix = prefix.expand(self.cls_num, -1, -1)

            suffix = suffix.expand(self.cls_num, -1, -1)

        

                                      

        prompts = torch.cat([prefix, clsn, suffix], dim=1)

        return prompts                                                           



    def get_learnable_token_span(self):

                                                          

        return self.learnable_start, self.learnable_end



    def slice_learnable_prompt_tokens(self, prompt_seq):

           

                                                          



                                              

                                      

           

        return prompt_seq[:, self.learnable_start:self.learnable_end, :]



class CustomCLIP(nn.Module):

       

                       



         

                                        

                                        

                                         

                                  

                                                                  

       

    def __init__(self, cfg, classnames, clip_model):

        super().__init__()

        self.cfg = cfg

        self.classnames = list(classnames)



                                                                              

        self.prompt_learner = ClsnameLearner(cfg, len(classnames), clip_model)           



                                                    

        self.tokenized_prompts = self.prompt_learner.tokenized_prompt



                          

        self.text_encoder = TextEncoder(clip_model)

        self.image_encoder = clip_model.visual                

        

        self.logit_scale = 20                                                          

        self.dtype = clip_model.dtype                                           

        self.clip_model = clip_model                                  

        subset_indices, subset_map = self._build_train_prompt_subset(self.classnames)                             

        if subset_indices is not None:                                                         

            self.register_buffer("train_prompt_subset_indices", subset_indices, persistent=False)

            self.register_buffer("train_prompt_subset_map", subset_map, persistent=False)

        else:

            self.train_prompt_subset_indices = None

            self.train_prompt_subset_map = None



    def _normalize_features(self, x): 

                                             

        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)



    def _build_train_prompt_subset(self, classnames):                                           

        filter_path = str(getattr(self.cfg.EXP, "TRAIN_CLASS_FILTER_JSON", "") or "").strip()

        if not filter_path:

            return None, None



        if not os.path.isabs(filter_path):

            filter_path = os.path.join(self.cfg.DATASET.TEXT_PATH, filter_path)

        with open(filter_path, "r", encoding="utf-8") as fp:

            payload = json.load(fp)



        if isinstance(payload, dict):

            raw_names = payload.keys()

        elif isinstance(payload, list):

            raw_names = payload

        else:

            raise ValueError("EXP.TRAIN_CLASS_FILTER_JSON must contain a dict or list of class names")



        normalize = lambda name: str(name).strip().replace("_", " ").lower()

        wanted = {normalize(name) for name in raw_names if str(name).strip()}

        subset = [idx for idx, name in enumerate(classnames) if normalize(name) in wanted]

        if len(subset) <= 0:

            raise ValueError("EXP.TRAIN_CLASS_FILTER_JSON did not match any model class names")



        subset_indices = torch.tensor(subset, dtype=torch.long)

        subset_map = torch.full((len(classnames),), -1, dtype=torch.long)

        subset_map[subset_indices] = torch.arange(len(subset), dtype=torch.long)

        return subset_indices, subset_map



    def get_eval_tta_info(self):                                              

           

                  



                                                                                                                

                         

           

        info = {

            "enabled": bool(getattr(self.cfg.TTA, "ENABLED", False)),

            "mode": "disabled",

            "scope": "all",

            "include_base_prompt": False,

            "aggregation": "none",

            "base_weight": 1.0,

            "template_weights": [],

            "num_templates": 0,

            "templates": [],

        }

        if not info["enabled"]:

            return info



        templates = list(getattr(self.cfg.TTA, "PROMPT_TEMPLATES", []))

        info["mode"] = getattr(self.cfg.TTA, "MODE", "multi_prompt")

        info["scope"] = getattr(self.cfg.TTA, "SCOPE", "all")

        info["include_base_prompt"] = bool(getattr(self.cfg.TTA, "INCLUDE_BASE_PROMPT", False))

        info["aggregation"] = getattr(self.cfg.TTA, "AGGREGATION", "feature_mean")

        info["base_weight"] = float(getattr(self.cfg.TTA, "BASE_WEIGHT", 1.0))

        info["template_weights"] = list(getattr(self.cfg.TTA, "TEMPLATE_WEIGHTS", []))

        info["num_templates"] = len(templates)

        info["templates"] = templates

        return info



    def get_eval_chunk_info(self):

        info = {

            "enabled": bool(getattr(self.cfg.CHUNK, "ENABLED", False)),

            "mode": "disabled",

            "scope": "all",

            "size": 0,

            "topk_per_chunk": 0,

            "filter_value": -1e4,

            "balanced": True,

        }

        if not info["enabled"]:

            return info



        info["mode"] = str(getattr(self.cfg.CHUNK, "MODE", "filter_topk"))

        info["scope"] = str(getattr(self.cfg.CHUNK, "SCOPE", "unseen_only"))

        info["size"] = int(getattr(self.cfg.CHUNK, "SIZE", 0))

        info["topk_per_chunk"] = int(getattr(self.cfg.CHUNK, "TOPK_PER_CHUNK", 0))

        info["filter_value"] = float(getattr(self.cfg.CHUNK, "FILTER_VALUE", -1e4))

        info["balanced"] = bool(getattr(self.cfg.CHUNK, "BALANCED", True))

        return info



    def get_eval_rerank_info(self):

        info = {

            "enabled": bool(getattr(self.cfg.RERANK, "ENABLED", False)),

            "mode": "disabled",

            "scope": "all",

            "topk": 0,

            "sim_temperature": 0.2,

            "logit_temperature": 1.0,

            "gate_mode": "none",

            "gate_topk": 0,

            "gate_threshold": 0.0,

        }

        if not info["enabled"]:

            return info



        info["mode"] = str(getattr(self.cfg.RERANK, "MODE", "proto_residual"))

        info["scope"] = str(getattr(self.cfg.RERANK, "SCOPE", "unseen_only"))

        info["topk"] = int(getattr(self.cfg.RERANK, "TOPK", 0))

        info["sim_temperature"] = float(getattr(self.cfg.RERANK, "SIM_TEMPERATURE", 0.2))

        info["logit_temperature"] = float(getattr(self.cfg.RERANK, "LOGIT_TEMPERATURE", 1.0))

        info["gate_mode"] = str(getattr(self.cfg.RERANK, "GATE_MODE", "none"))

        info["gate_topk"] = int(getattr(self.cfg.RERANK, "GATE_TOPK", 0))

        info["gate_threshold"] = float(getattr(self.cfg.RERANK, "GATE_THRESHOLD", 0.0))

        return info



    def _build_label_chunk_ranges(self, total_count, chunk_size, balanced=True):

        if total_count <= 0:

            return []

        if chunk_size <= 0 or chunk_size >= total_count:

            return [(0, total_count)]



        if not balanced:

            ranges = []

            start = 0

            while start < total_count:

                end = min(total_count, start + chunk_size)

                ranges.append((start, end))

                start = end

            return ranges



        num_chunks = math.ceil(total_count / chunk_size)

        base_chunk = total_count // num_chunks

        remainder = total_count % num_chunks



        ranges = []

        start = 0

        for chunk_idx in range(num_chunks):

            cur_size = base_chunk + (1 if chunk_idx < remainder else 0)

            end = start + cur_size

            ranges.append((start, end))

            start = end

        return ranges



    def apply_eval_label_chunking(self, logits, num_seen_classes=None):

        chunk_info = self.get_eval_chunk_info()

        if not chunk_info["enabled"]:

            return logits

        if logits.dim() != 2:

            raise ValueError(f"Chunked label competition expects 2D logits, got shape {tuple(logits.shape)}")



        mode = chunk_info["mode"]

        if mode != "filter_topk":

            raise ValueError(f"Unsupported CHUNK.MODE: {mode}")



        scope = chunk_info["scope"]

        total_classes = logits.shape[1]

        if scope == "all":

            target_offset = 0

            target_count = total_classes

        elif scope == "unseen_only":

            if num_seen_classes is None:

                raise ValueError("CHUNK.SCOPE='unseen_only' requires num_seen_classes")

            if num_seen_classes < 0 or num_seen_classes > total_classes:

                raise ValueError(

                    f"num_seen_classes must be within [0, {total_classes}], got {num_seen_classes}"

                )

            target_offset = int(num_seen_classes)

            target_count = total_classes - target_offset

            if target_count <= 0:

                return logits

        else:

            raise ValueError(f"Unsupported CHUNK.SCOPE: {scope}")



        chunk_size = int(chunk_info["size"])

        topk_per_chunk = int(chunk_info["topk_per_chunk"])

        if topk_per_chunk <= 0:

            raise ValueError(f"CHUNK.TOPK_PER_CHUNK must be positive, got {topk_per_chunk}")



        chunk_ranges = self._build_label_chunk_ranges(

            target_count,

            chunk_size=chunk_size,

            balanced=bool(chunk_info["balanced"]),

        )

        if len(chunk_ranges) <= 1:

            return logits



        keep_mask = torch.zeros_like(logits, dtype=torch.bool)

        if scope == "unseen_only" and target_offset > 0:

            keep_mask[:, :target_offset] = True



        for start, end in chunk_ranges:

            chunk_logits = logits[:, target_offset + start : target_offset + end]

            cur_topk = min(topk_per_chunk, chunk_logits.shape[1])

            if cur_topk >= chunk_logits.shape[1]:

                keep_mask[:, target_offset + start : target_offset + end] = True

                continue



            topk_indices = torch.topk(chunk_logits, k=cur_topk, dim=1).indices

            chunk_keep = torch.zeros_like(chunk_logits, dtype=torch.bool)

            chunk_keep.scatter_(1, topk_indices, True)

            keep_mask[:, target_offset + start : target_offset + end] = chunk_keep



        filter_value = float(chunk_info["filter_value"])

        return logits.masked_fill(~keep_mask, filter_value)



    def apply_eval_proto_rerank(self, logits, text_features, num_seen_classes=None):

        rerank_info = self.get_eval_rerank_info()              

        if not rerank_info["enabled"]:                           

            return logits

        if logits.dim() != 2:                                                                                                                            

            raise ValueError(f"Prototype rerank expects 2D logits, got shape {tuple(logits.shape)}")

        if text_features.dim() != 2:

            raise ValueError(

                f"Prototype rerank expects 2D text_features, got shape {tuple(text_features.shape)}"

            )

        if logits.shape[1] != text_features.shape[0]:

            raise ValueError(

                "Prototype rerank requires logits/classes to match text_features/classes, "

                f"got {logits.shape[1]} and {text_features.shape[0]}"

            )



        mode = rerank_info["mode"]                                                               

        supported_modes = {"proto_residual", "proto_residual_positive_only", "proto_positive_only"}

        if mode not in supported_modes:

            raise ValueError(f"Unsupported RERANK.MODE: {mode}")



        scope = rerank_info["scope"]                          

        total_classes = logits.shape[1]  

        if scope == "all":     

            target_offset = 0

            target_count = total_classes

        elif scope == "unseen_only":           

            if num_seen_classes is None:

                raise ValueError("RERANK.SCOPE='unseen_only' requires num_seen_classes")

            if num_seen_classes < 0 or num_seen_classes > total_classes:

                raise ValueError(

                    f"num_seen_classes must be within [0, {total_classes}], got {num_seen_classes}"

                )

            target_offset = int(num_seen_classes)                        

            target_count = total_classes - target_offset                                            

            if target_count <= 1:

                return logits

        else:

            raise ValueError(f"Unsupported RERANK.SCOPE: {scope}")



        topk = int(rerank_info["topk"])

        if topk <= 0:      

            return logits

        topk = min(topk, target_count - 1)

        if topk <= 0:

            return logits



        sim_temperature = float(rerank_info["sim_temperature"])      

        logit_temperature = float(rerank_info["logit_temperature"])

        if sim_temperature <= 0:       

            raise ValueError(f"RERANK.SIM_TEMPERATURE must be positive, got {sim_temperature}")

        if logit_temperature <= 0:

            raise ValueError(f"RERANK.LOGIT_TEMPERATURE must be positive, got {logit_temperature}")



        target_logits = logits[:, target_offset : target_offset + target_count]                                  

        target_text = text_features[target_offset : target_offset + target_count]           

        target_text = self._normalize_features(target_text.float())                       

        gate_mode = str(rerank_info.get("gate_mode", "none")).lower()                     

        gate_threshold = float(rerank_info.get("gate_threshold", 0.0))

        gate_topk = int(rerank_info.get("gate_topk", 0))



        if gate_mode in {"none", "off", "disabled"}:

            gate_mask = None

        elif gate_mode == "logit_gt":

            gate_mask = target_logits > gate_threshold

        elif gate_mode == "topk":

            if gate_topk <= 0:

                raise ValueError(f"RERANK.GATE_TOPK must be positive for gate_mode='topk', got {gate_topk}")

            gate_topk = min(gate_topk, target_count)

            gate_mask = torch.zeros_like(target_logits, dtype=torch.bool)

            gate_idx = torch.topk(target_logits, k=gate_topk, dim=1).indices

            gate_mask.scatter_(1, gate_idx, True)

        else:

            raise ValueError(f"Unsupported RERANK.GATE_MODE: {gate_mode}")



        proto_sim = target_text @ target_text.t()                                                                                           

        proto_sim.fill_diagonal_(-1e4)

        neighbor_sim, neighbor_idx = torch.topk(proto_sim, k=topk, dim=1)                                          

        neighbor_weights = F.softmax(neighbor_sim / sim_temperature, dim=1)                                                        



        self_idx = torch.arange(target_count, device=logits.device).unsqueeze(1)

        local_idx = torch.cat([self_idx, neighbor_idx], dim=1)                      



        expanded_logits = target_logits.unsqueeze(1).expand(-1, target_count, -1)

        gather_idx = local_idx.unsqueeze(0).expand(target_logits.shape[0], -1, -1)

        local_scores = torch.gather(expanded_logits, dim=2, index=gather_idx)                                                                         

        local_probs = F.softmax(local_scores / logit_temperature, dim=2)                                               



        self_prob = local_probs[:, :, 0]                               

        competitor_prob = (                                                 

            local_probs[:, :, 1:] * neighbor_weights.unsqueeze(0).to(dtype=local_probs.dtype)

        ).sum(dim=2)

        local_span = local_scores.max(dim=2).values - local_scores.min(dim=2).values                          

        residual = (self_prob - competitor_prob) * local_span                                                                              

        if mode in {"proto_residual_positive_only", "proto_positive_only"}:

            residual = torch.clamp(residual, min=0.0)                                                                                  

        if gate_mask is not None:      

            residual = residual.masked_fill(~gate_mask, 0.0)



        reranked_logits = logits.clone()

        reranked_logits[:, target_offset : target_offset + target_count] = (

            target_logits + residual.to(dtype=target_logits.dtype)                                                                             

        )

        return reranked_logits                           



    def _build_template_prompts(self, template):

           

                                         



                                        

                                        

           

        if template.count("{}") != 1:

            raise ValueError(

                "TTA prompt templates must contain exactly one '{}' placeholder, "

                f"got: {template}"

            )



                                 

        prompt_learner = self.prompt_learner



                             

                                              

        placeholder = " ".join(["X"] * prompt_learner.n_clsn)

        prompt_text = template.replace("_", " ").format(placeholder)

        prefix_text, _ = template.replace("_", " ").split("{}", 1)

        prefix_len = 1 + len(_tokenizer.encode(prefix_text))



        tokenized_prompt = clip.tokenize(prompt_text)

        class_tokens = prompt_learner.get_class_tokens()

        device = class_tokens.device

        tokenized_prompt = tokenized_prompt.to(device)



                                    

        with torch.no_grad():

            embedding = self.text_encoder.token_embedding(tokenized_prompt).type(self.dtype)



                                                   

        token_prefix = embedding[:, :prefix_len, :]

        token_suffix = embedding[:, prefix_len + prompt_learner.n_clsn :, :]



                          

        if token_prefix.shape[0] == 1:

            token_prefix = token_prefix.expand(prompt_learner.cls_num, -1, -1)

            token_suffix = token_suffix.expand(prompt_learner.cls_num, -1, -1)



        prompts = torch.cat([token_prefix, class_tokens, token_suffix], dim=1)

        return prompts, tokenized_prompt



    def encode_prompt_features_from_template(self, template, return_sequence=False):

        prompts, tokenized_prompts = self._build_template_prompts(template)

        if return_sequence:

            cls_embeddings, prompt_seq = self.text_encoder(

                prompts, tokenized_prompts, return_sequence=True

            )

            cls_embeddings = self._normalize_features(cls_embeddings)

            return cls_embeddings, prompt_seq



        cls_embeddings = self.text_encoder(prompts, tokenized_prompts)

        cls_embeddings = self._normalize_features(cls_embeddings)

        return cls_embeddings



    def encode_visual_features(self, image):

        image_features, image_features_ori, attn_weights = self.image_encoder(image.type(self.dtype))                                    



        image_features_ = self._normalize_features(image_features_ori)

        image_features = self._normalize_features(image_features)

        aux_globals = {}

        cls_global = getattr(self.image_encoder, "last_cls_global_feature", None)

        if cls_global is not None:

            aux_globals["cls_global"] = self._normalize_features(cls_global)

        last_global = getattr(self.image_encoder, "last_last_global_feature", None)

        if last_global is not None:

            aux_globals["last_global"] = self._normalize_features(last_global)



        return image_features, image_features_, attn_weights, aux_globals



    def _apply_last_unseen_rescue(self, logit_glob_cls, logit_glob_last, num_seen_classes):

        if num_seen_classes is None or logit_glob_last is None:

            return logit_glob_cls



        if num_seen_classes <= 0 or num_seen_classes >= logit_glob_cls.shape[1]:

            return logit_glob_cls



        logit_glob = logit_glob_cls.clone()

        logit_glob[:, num_seen_classes:] = torch.maximum(

            logit_glob_cls[:, num_seen_classes:],

            logit_glob_last[:, num_seen_classes:],

        )

        return logit_glob



    def encode_prompt_features(self, return_sequence=False, class_indices=None):

        prompts = self.prompt_learner(class_indices=class_indices)                                                                                      

        tokenized_prompts = self.tokenized_prompts                                                                                                                               

        if return_sequence:      

            cls_embeddings, prompt_seq = self.text_encoder(

                prompts, tokenized_prompts, return_sequence=True

            )

            cls_embeddings = self._normalize_features(cls_embeddings)

            return cls_embeddings, prompt_seq



        cls_embeddings = self.text_encoder(prompts, tokenized_prompts)             

        cls_embeddings = self._normalize_features(cls_embeddings)                                   

        return cls_embeddings                           



    def _build_tta_weights(self, tta_info, feature_count):

        aggregation = tta_info["aggregation"]

        if aggregation == "feature_mean":        

            return None

        if aggregation != "weighted_feature_mean":  

            raise ValueError(f"Unsupported TTA.AGGREGATION: {aggregation}")



        weights = []                           

        if tta_info["include_base_prompt"]:

            weights.append(float(tta_info["base_weight"]))



        template_weights = tta_info["template_weights"]

        if len(template_weights) == 0:       

            template_weights = [1.0] * tta_info["num_templates"]

        elif len(template_weights) != tta_info["num_templates"]:        

            raise ValueError(

                "TTA.TEMPLATE_WEIGHTS length must match TTA.PROMPT_TEMPLATES length, "

                f"got {len(template_weights)} and {tta_info['num_templates']}"

            )

        weights.extend(float(weight) for weight in template_weights)



        if len(weights) != feature_count:

            raise ValueError(

                f"TTA weight count mismatch for {feature_count} features, got {len(weights)} weights"

            )



        weights = torch.tensor(weights, dtype=torch.float32, device=self.prompt_learner.clsn.device)

        if torch.any(weights <= 0):

            raise ValueError("All TTA weights must be positive")

        return weights / weights.sum().clamp_min(1e-12)



    def _aggregate_tta_features(self, feature_list, stage_name, tta_info):

        if len(feature_list) <= 0:

            raise ValueError(f"No features were provided for {stage_name}")

        if len(feature_list) == 1:

            return self._normalize_features(feature_list[0])



        stacked = torch.stack(feature_list, dim=0)

        weights = self._build_tta_weights(tta_info, feature_count=stacked.shape[0])

        if weights is None:

            stacked = stacked.mean(dim=0)

        else:

            weights = weights.to(dtype=stacked.dtype).view(-1, 1, 1)

            stacked = (stacked * weights).sum(dim=0)

        return self._normalize_features(stacked)

    

    def encode_text_features(self, use_tta=None, num_seen_classes=None):                                

        if use_tta is None:                                   

            use_tta = bool(getattr(self.cfg.TTA, "ENABLED", False))

        if not use_tta:                                                                                        

            return self.encode_prompt_features()



        tta_info = self.get_eval_tta_info()                                                                                                                                                           

        if not tta_info["enabled"]:                                                                                                                                         

            return self.encode_prompt_features()

        if tta_info["mode"] != "multi_prompt":

            raise ValueError(f"Unsupported TTA.MODE: {tta_info['mode']}")

        if tta_info["aggregation"] not in {"feature_mean", "weighted_feature_mean"}:

            raise ValueError(f"Unsupported TTA.AGGREGATION: {tta_info['aggregation']}")

        if tta_info["scope"] not in {"all", "unseen_only"}:

            raise ValueError(f"Unsupported TTA.SCOPE: {tta_info['scope']}")

        if tta_info["num_templates"] <= 0 and not tta_info["include_base_prompt"]:

            raise ValueError(

                "TTA.ENABLED=True requires at least one prompt template or INCLUDE_BASE_PROMPT=True"

            )



        if tta_info["scope"] == "all":             

            text_features = []

            if tta_info["include_base_prompt"]:

                text_features.append(self.encode_prompt_features())

            text_features.extend(

                self.encode_prompt_features_from_template(template)

                for template in tta_info["templates"]

            )

            return self._aggregate_tta_features(

                text_features,

                stage_name="global TTA aggregation",

                tta_info=tta_info,

            )



        base_text_features = self.encode_prompt_features()                                                                   

        if num_seen_classes is None:                                   

            raise ValueError("TTA.SCOPE='unseen_only' requires num_seen_classes")

        if num_seen_classes < 0 or num_seen_classes > base_text_features.shape[0]:

            raise ValueError(

                "num_seen_classes must be within [0, total_classes], "

                f"got {num_seen_classes} for {base_text_features.shape[0]} classes"

            )

        if num_seen_classes >= base_text_features.shape[0]:

            return base_text_features                                 



        unseen_features = []

        if tta_info["include_base_prompt"]:

            unseen_features.append(base_text_features[num_seen_classes:])

        unseen_features.extend(                                                                       

            self.encode_prompt_features_from_template(template)[num_seen_classes:]

            for template in tta_info["templates"]

        )

        unseen_features = self._aggregate_tta_features(               

            unseen_features,

            stage_name="unseen-only TTA aggregation",

            tta_info=tta_info,

        )



        if num_seen_classes <= 0:           

            return unseen_features



        text_features = torch.cat(                                                                                        

            [base_text_features[:num_seen_classes], unseen_features],

            dim=0,

        )

        return self._normalize_features(text_features)                                                            



    def build_ot_aligned_text_features(      

        self,

        text_features,

        caption_seq,

        caption_mask,

        target_prompt_parts,

        ot_runtime=None,

    ):

        if self.cfg.OT.SLOT_POOL != "cls_guided":

            raise ValueError(f"Unsupported OT.SLOT_POOL: {self.cfg.OT.SLOT_POOL}")



        if self.cfg.OT.FP32:                                           

            caption_seq_ot = caption_seq.float()

            target_prompt_parts_ot = target_prompt_parts.float()

            caption_mask_ot = caption_mask.float()

        else:

            caption_seq_ot = caption_seq

            target_prompt_parts_ot = target_prompt_parts

            caption_mask_ot = caption_mask



        cost = cosine_cost_matrix(caption_seq_ot, target_prompt_parts_ot)                                      

        src_mass = None

        src_mass_mode = self.cfg.OT.SRC_MASS

        if ot_runtime is not None:

            src_mass_mode = ot_runtime.get("src_mass", src_mass_mode)

        if src_mass_mode == "uniform":

            pass

        elif src_mass_mode == "ram_softmax":

            sim = 1.0 - cost

            src_mass = build_similarity_source_mass(

                sim=sim,

                src_mask=caption_mask_ot,

                temperature=float(self.cfg.OT.SRC_MASS_TEMP),

                detach=bool(self.cfg.OT.SRC_MASS_DETACH),

            )

        else:

            raise ValueError(f"Unsupported OT.SRC_MASS: {src_mass_mode}")



        transport, tgt_mass = compute_ot_transport(

            cost=cost,

            src_mask=caption_mask_ot,

            src_mass=src_mass,

            eps=self.cfg.OT.EPS,

            max_iter=self.cfg.OT.MAX_ITER,

        )

        aligned_caption_parts = transport_barycentric_projection(

            src_seq=caption_seq_ot,

            transport=transport,

            tgt_mass=tgt_mass,

            src_mask=caption_mask_ot,

        )



        slot_logits = F.cosine_similarity(

            aligned_caption_parts,

            target_prompt_parts_ot,

            dim=-1,

        )

        slot_weights = F.softmax(slot_logits, dim=-1)

        aligned_text_features = torch.sum(

            aligned_caption_parts * slot_weights.unsqueeze(-1),

            dim=1,

        )

        aligned_text_features = self._normalize_features(aligned_text_features)



        blend = float(self.cfg.OT.ALIGN_BLEND)

        if ot_runtime is not None:

            blend = float(ot_runtime.get("align_blend", blend))

        fused_text_features = (1.0 - blend) * text_features.float() + blend * aligned_text_features

        fused_text_features = self._normalize_features(fused_text_features)



        output_dtype = text_features.dtype

        outputs = {

            "text_features": fused_text_features.to(dtype=output_dtype),

            "aligned_text_features": aligned_text_features.to(dtype=output_dtype),

            "aligned_caption_parts": aligned_caption_parts.to(dtype=output_dtype),

            "ot_transport": transport.to(dtype=output_dtype),

            "ot_slot_weights": slot_weights.to(dtype=output_dtype),

            "ot_align_blend": torch.tensor(blend, device=text_features.device, dtype=output_dtype),

        }

        if src_mass is not None:

            outputs["ot_src_mass"] = src_mass.to(dtype=output_dtype)

        return outputs





                                                                

    def forward(self, caption, targets=None, return_ot=False, ot_runtime=None):

        if return_ot and targets is None:

            raise ValueError("targets must be provided when return_ot=True")



        prompt_class_indices = None

        local_targets = targets

        if self.training and targets is not None and self.train_prompt_subset_indices is not None:

            prompt_class_indices = self.train_prompt_subset_indices

            local_targets = self.train_prompt_subset_map[targets]

            if torch.any(local_targets < 0):

                raise ValueError("Batch targets contain classes outside EXP.TRAIN_CLASS_FILTER_JSON")

        if (

            self.training

            and targets is not None

            and bool(getattr(self.cfg.EXP, "TRAIN_BATCH_CLASS_SUBSET", False))

            and local_targets is not None

            and local_targets.numel() > 0

        ):

            batch_unique_targets = torch.unique(local_targets.detach(), sorted=True)

            if batch_unique_targets.numel() > 0:

                if prompt_class_indices is None:

                    prompt_class_indices = batch_unique_targets

                else:

                    prompt_class_indices = prompt_class_indices.index_select(0, batch_unique_targets)

                local_subset_map = torch.full(

                    (int(local_targets.max().item()) + 1,),

                    -1,

                    dtype=torch.long,

                    device=local_targets.device,

                )

                local_subset_map[batch_unique_targets] = torch.arange(

                    batch_unique_targets.numel(),

                    dtype=torch.long,

                    device=local_targets.device,

                )

                local_targets = local_subset_map[local_targets]

                if torch.any(local_targets < 0):

                    raise ValueError("Failed to remap local targets for TRAIN_BATCH_CLASS_SUBSET")



        if return_ot:       

            text_features, caption_seq = self.text_encoder(                                                  

                caption, None, if_embedding=False, return_sequence=True

            )

            cls_embeddings, prompt_seq = self.encode_prompt_features(                                                   

                return_sequence=True,

                class_indices=prompt_class_indices,

            )



            text_features = self._normalize_features(text_features)

            caption_mask = build_caption_mask(caption)

            prompt_parts = self.prompt_learner.slice_learnable_prompt_tokens(prompt_seq)

            target_prompt_parts = prompt_parts[local_targets]



            outputs = {

                "text_features": text_features,

                "text_features_raw": text_features,

                "cls_embeddings": cls_embeddings,

                "caption_seq": caption_seq,

                "caption_mask": caption_mask,

                "caption_token_ids": caption,

                "prompt_seq": prompt_seq,

                "prompt_parts": prompt_parts,

                "target_prompt_parts": target_prompt_parts,

                "local_targets": local_targets,

            }

            ot_mode = self.cfg.OT.MODE if ot_runtime is None else ot_runtime.get("mode", self.cfg.OT.MODE)

            if ot_mode == "forward_align":                                  

                outputs.update(

                    self.build_ot_aligned_text_features(

                        text_features=text_features,

                        caption_seq=caption_seq,

                        caption_mask=caption_mask,

                        target_prompt_parts=target_prompt_parts,

                        ot_runtime=ot_runtime,

                    )

                )



            return outputs

             

        text_features = self.text_encoder(caption, None, if_embedding=False)

        cls_embeddings = self.encode_prompt_features(class_indices=prompt_class_indices)



        text_features = self._normalize_features(text_features)



        if local_targets is not None and prompt_class_indices is not None:

            return {

                "text_features": text_features,

                "cls_embeddings": cls_embeddings,

                "local_targets": local_targets,

            }



        return text_features, cls_embeddings

    

    def forward_for_open(self, image, text_features, num_seen_classes=None, return_rerank_bundle=False):

        logit_scale = self.logit_scale

        image_features, image_features_, attn_weights, aux_globals = self.encode_visual_features(image)

        logit_local = logit_scale * image_features[:, 1:, :] @ text_features.t()

        logit_local = logit_local.transpose(1, 2)



        w_avg = F.softmax(logit_local, dim=2)

        logit_local_pool = torch.sum(logit_local * w_avg, dim=2)        

        

        logit_glob = logit_scale * image_features_ @ text_features.t()

        if str(getattr(self.cfg.LAST, "MODE", "off")).lower() == "unseen_rescue":

            cls_global = aux_globals.get("cls_global")

            last_global = aux_globals.get("last_global")

            if cls_global is not None and last_global is not None:

                logit_glob_cls = logit_scale * cls_global @ text_features.t()

                logit_glob_last = logit_scale * last_global @ text_features.t()

                logit_glob = self._apply_last_unseen_rescue(logit_glob_cls, logit_glob_last, num_seen_classes)

        logit_final = 0.5 * (logit_glob + logit_local_pool)



        if return_rerank_bundle:

            return {

                "logit_final": logit_final,

                "logit_glob": logit_glob,

                "logit_local": logit_local,

                "logit_local_pool": logit_local_pool,

                "visual_feature": image_features_,

            }

        return logit_final

    

    def forward_for_test(self, image, text_features, num_seen_classes=None, return_rerank_bundle=False):

        image_features, image_features_, attn_weights, aux_globals = self.encode_visual_features(image)                                                                                                          



        logit_local = image_features[:, 1:, :] @ text_features.t()                              

        logit_local = logit_local.transpose(1, 2)                       

        _, C, HW = logit_local.shape            

        H = int(math.sqrt(HW))           

        logit_local = logit_local.view(-1, C, H, H)                                                        

        

        logit_glob = image_features_ @ text_features.t()                        

        if str(getattr(self.cfg.LAST, "MODE", "off")).lower() == "unseen_rescue":      

            cls_global = aux_globals.get("cls_global")

            last_global = aux_globals.get("last_global")

            if cls_global is not None and last_global is not None:

                logit_glob_cls = cls_global @ text_features.t()

                logit_glob_last = last_global @ text_features.t()

                logit_glob = self._apply_last_unseen_rescue(logit_glob_cls, logit_glob_last, num_seen_classes)



        if return_rerank_bundle:                                         

            return {

                "logit_local_map": logit_local,                    

                "logit_glob": logit_glob,                   

                "visual_feature": image_features_,                                    

            }

        return logit_local, logit_glob



    def aggregatorplus(self, x, x_list, count_list=[2,3], return_rerank_bundle=False):

        logit_scale = self.logit_scale     

        visual_feature = None

        if return_rerank_bundle:        

            logit_local, logit_glob = x["logit_local_map"], x["logit_glob"]

            visual_feature = x["visual_feature"]

        else:

            logit_local, logit_glob = x

        list_local, list_glob = list(zip(*x_list))                                                                                                         

        multiscale_list_local = []      

        multiscale_list_glob = []

        for count in count_list:                                             

            tmp_scale_l = list_local[: count*count]                              

            tmp_scale_g = list_glob[: count*count]

            img_cat = []

            for i in range(count):  

                img_cat.append(torch.cat(tmp_scale_l[i*count:(i+1)*count], dim=-1))                             

            img_cat = torch.cat(img_cat, dim=-2)                   

            tmp_local = F.max_pool2d(img_cat, count, count)                                                             

            multiscale_list_local.append(tmp_local)



            tmp_glob = torch.stack(tmp_scale_g, dim=0)

            tmp_glob = tmp_glob.max(dim=0)[0]                            

            multiscale_list_glob.append(tmp_glob)                      

            list_local = list_local[count*count:]                                                     

            list_glob = list_glob[count*count:]



        multiscale_list_local.append(logit_local)

        logit_local = torch.stack(multiscale_list_local, dim=0)                          

        logit_local = logit_local.mean(dim=0)

        logit_local = logit_local.view(logit_local.size(0), logit_local.size(1), -1)                               

        logit_local = logit_scale * logit_local       

        w_avg = F.softmax(logit_local, dim=2)                                              

        logit_local_pool = torch.sum(logit_local * w_avg, dim=2)                                            



        multiscale_list_glob.append(logit_glob) 

        logit_glob = torch.stack(multiscale_list_glob, dim=0)

        logit_glob = logit_glob.mean(dim=0)                                                       

        logit_glob = logit_glob * logit_scale                   



        logit_final = 0.5 * (logit_glob + logit_local_pool)                    



        if return_rerank_bundle:                                  

            return {

                "logit_final": logit_final,

                "logit_glob": logit_glob,

                "logit_local": logit_local,

                "logit_local_pool": logit_local_pool,

                "visual_feature": visual_feature,

            }

        return logit_final





def build_text_model(cfg, classnames):                                                                              

    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")

    clip_model = load_clip_to_cpu(cfg)            

    clip_model.float()

    

    model = CustomCLIP(cfg, classnames, clip_model)                                    



    for name, param in model.named_parameters():

        if "prompt_learner" not in name:

            param.requires_grad_(False)



    if torch.cuda.is_available() and cfg.USE_CUDA:

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    model.to(device)



    return model

