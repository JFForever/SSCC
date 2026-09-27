from collections import OrderedDict

from typing import Tuple, Union



import numpy as np

import torch

import torch.nn.functional as F

from torch import nn





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1):

        super().__init__()



                                                                                                             

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)



        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)



        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()



        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * self.expansion)



        self.relu = nn.ReLU(inplace=True)

        self.downsample = None

        self.stride = stride



        if stride > 1 or inplanes != planes * Bottleneck.expansion:

                                                                                                          

            self.downsample = nn.Sequential(OrderedDict([

                ("-1", nn.AvgPool2d(stride)),

                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),

                ("1", nn.BatchNorm2d(planes * self.expansion))

            ]))



    def forward(self, x: torch.Tensor):

        identity = x



        out = self.relu(self.bn1(self.conv1(x)))

        out = self.relu(self.bn2(self.conv2(out)))

        out = self.avgpool(out)

        out = self.bn3(self.conv3(out))



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)

        return out





class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):

        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)

        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)

        self.num_heads = num_heads



    def forward(self, x):

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)                  

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)            

        x = x + self.positional_embedding[:, None, :].to(x.dtype)            

        x, _ = F.multi_head_attention_forward(

            query=x, key=x, value=x,

            embed_dim_to_check=x.shape[-1],

            num_heads=self.num_heads,

            q_proj_weight=self.q_proj.weight,

            k_proj_weight=self.k_proj.weight,

            v_proj_weight=self.v_proj.weight,

            in_proj_weight=None,

            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),

            bias_k=None,

            bias_v=None,

            add_zero_attn=False,

            dropout_p=0,

            out_proj_weight=self.c_proj.weight,

            out_proj_bias=self.c_proj.bias,

            use_separate_proj_weight=True,

            training=self.training,

            need_weights=False

        )



        return x[0]





class ModifiedResNet(nn.Module):

       

                                                                                       

                                                                                                      

                                                                                                                

                                                                           

       



    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):

        super().__init__()

        self.output_dim = output_dim

        self.input_resolution = input_resolution



                          

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(width // 2)

        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(width // 2)

        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(width)

        self.avgpool = nn.AvgPool2d(2)

        self.relu = nn.ReLU(inplace=True)



                         

        self._inplanes = width                                                         

        self.layer1 = self._make_layer(width, layers[0])

        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)

        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)

        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)



        embed_dim = width * 32                                

        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)



    def _make_layer(self, planes, blocks, stride=1):

        layers = [Bottleneck(self._inplanes, planes, stride)]



        self._inplanes = planes * Bottleneck.expansion

        for _ in range(1, blocks):

            layers.append(Bottleneck(self._inplanes, planes))



        return nn.Sequential(*layers)



    def forward(self, x):

        def stem(x):

            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:

                x = self.relu(bn(conv(x)))

            x = self.avgpool(x)

            return x



        x = x.type(self.conv1.weight.dtype)

        x = stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.attnpool(x)



        return x





class LayerNorm(nn.LayerNorm):

                                                    



    def forward(self, x: torch.Tensor):

        orig_type = x.dtype

        ret = super().forward(x.type(torch.float32))

        return ret.type(orig_type)





class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):

        return x * torch.sigmoid(1.702 * x)





class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):

        super().__init__()



        self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([

            ("c_fc", nn.Linear(d_model, d_model * 4)),

            ("gelu", QuickGELU()),

            ("c_proj", nn.Linear(d_model * 4, d_model))

        ]))

        self.ln_2 = LayerNorm(d_model)

        self.attn_mask = attn_mask



    def attention(self, x: torch.Tensor):

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]



    def forward(self, x: torch.Tensor):

        x = x + self.attention(self.ln_1(x))

        x = x + self.mlp(self.ln_2(x))

        return x





class ResidualAttentionBlock_IVLP(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,

                 text_layer=False, i=0, need_weights=True, design_details=None):

        super().__init__()



        self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([

            ("c_fc", nn.Linear(d_model, d_model * 4)),

            ("gelu", QuickGELU()),

            ("c_proj", nn.Linear(d_model * 4, d_model))

        ]))

        self.ln_2 = LayerNorm(d_model)

                                                       

                                                                               

                                                                              

                               

        self.text_layer = text_layer

        self.attn_mask = attn_mask

        self.need_weights = need_weights

        self.layer = i

        if i != 0:

            self.add_prompt = add_prompt

            if self.add_prompt:

                if self.text_layer:

                    self.n_ctx_text = design_details["language_ctx"]                  

                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)

                else:

                    self.n_ctx_visual = design_details["vision_ctx"]                  

                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)

                                                           

                nn.init.normal_(ctx_vectors, std=0.02)

                self.VPT_shallow = nn.Parameter(ctx_vectors)

        else:

            self.add_prompt = False



    def attention(self, x: torch.Tensor):

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x, x, need_weights=self.need_weights, attn_mask=self.attn_mask, average_attn_weights=True)



    def forward(self, x: torch.Tensor):

                                                                      

                                                     

        if self.add_prompt:

                                                                  

            if not self.text_layer:

                                                                                   

                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]

                                                                 

                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                                                                                                  

                                        

                x = torch.cat([prefix, visual_context], dim=0)

            else:

                                                                 

                                      

                                                                       

                prefix = x[:1, :, :]

                suffix = x[1 + self.n_ctx_text:, :, :]

                                                                 

                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                                                                                             

                                        

                x = torch.cat([prefix, textual_context, suffix], dim=0)

                

        if self.need_weights == False:

            x = x + self.attention(self.ln_1(x))[0]

            x = x + self.mlp(self.ln_2(x))

            return x

        else:

            x1, attn_w = self.attention(self.ln_1(x))

            x = x + x1

            x = x + self.mlp(self.ln_2(x))

            return x, attn_w





class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,

                 text_layer=False, need_weights=False, design_details=None):

        super().__init__()

        self.width = width

        self.layers = layers

        self.need_weights = need_weights

                                                                        

        current_trainer = design_details['trainer']

        if current_trainer == 'IVLP' or current_trainer == 'VPT':

            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True, text_layer, i, need_weights, design_details) if prompts_needed > i

                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False, text_layer, i, need_weights, design_details)

                                             for i in range(layers)])

        else:

            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):

        return self.resblocks(x)





class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, design_details, need_weights: bool = True):

        super().__init__()

        self.input_resolution = input_resolution

        self.output_dim = output_dim

        self.width = width

        self.heads = heads

        self.need_weights = need_weights

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        if design_details["vision_depth"] == 0:

            self.VPT_shallow = False

        else:

            self.VPT_shallow = True

        if self.VPT_shallow:

                                           

            n_ctx = design_details["vision_ctx"]                  

            ctx_vectors = torch.empty(n_ctx, width)

            nn.init.normal_(ctx_vectors, std=0.02)

            self.VPT = nn.Parameter(ctx_vectors)



        scale = width ** -0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        self.ln_pre = LayerNorm(width)

                                                                              

                                      

        self.prompt_till_layer_visual = design_details["vision_depth"]

        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual, need_weights=need_weights, design_details=design_details)



        self.ln_post = LayerNorm(width)

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.design_details = design_details

        self.n_ctx = design_details["vision_ctx"]

        self.last_pool_enabled = bool(design_details.get("last_enabled", False))

        self.last_pool_mode = self._normalize_last_pool_mode(design_details.get("last_mode", "replace"))

        self.last_pool_topk = int(design_details.get("last_topk", 0))

        self.last_pool_topk_ratio = float(design_details.get("last_topk_ratio", 0.5))

        self.last_pool_fusion_alpha = float(design_details.get("last_fusion_alpha", 0.1))

        self.last_pool_sigma = float(design_details.get("last_sigma", 0.0))

        self.last_pool_eps = float(design_details.get("last_eps", 1e-6))

        self._last_kernel_cache = None

        self._last_kernel_cache_spec = None

        

    def upsample_pos_emb(self, emb, new_size):

                                                                 

                      

        first = emb[:1, :]

        emb = emb[1:, :]

        N, D = emb.size(0), emb.size(1)

        size = int(np.sqrt(N))

        assert size * size == N

                                        

        emb = emb.permute(1, 0)

        emb = emb.view(1, D, size, size).contiguous()

        emb = F.upsample(emb, size=new_size, mode='bilinear',)

        emb = emb.view(D, -1).contiguous()

        emb = emb.permute(1, 0)

        emb = torch.cat([first, emb], 0)

        emb = nn.parameter.Parameter(emb.half())

        return emb 



    def _normalize_last_pool_mode(self, mode):

        mode = str(mode).lower()

        if mode not in {"off", "replace", "blend", "consensus", "unseen_rescue"}:

            return "replace"

        return mode



    def _consensus_last_fusion(self, cls_global: torch.Tensor, last_global: torch.Tensor):

        cls_norm = F.normalize(cls_global.float(), dim=-1)

        last_norm = F.normalize(last_global.float(), dim=-1)

        agreement = (cls_norm * last_norm).sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)

                                                                                                       

        gate = agreement * (1.0 - agreement)

        return cls_global + gate.to(dtype=cls_global.dtype) * (last_global - cls_global)



    def set_last_pool_config(self, enabled=False, mode="replace", topk=0, topk_ratio=0.5, fusion_alpha=0.1, sigma=0.0, eps=1e-6):

        self.last_pool_enabled = bool(enabled)

        self.last_pool_mode = self._normalize_last_pool_mode(mode)

        self.last_pool_topk = int(topk)

        self.last_pool_topk_ratio = float(topk_ratio)

        self.last_pool_fusion_alpha = float(fusion_alpha)

        self.last_pool_sigma = float(sigma)

        self.last_pool_eps = float(eps)

        self._last_kernel_cache = None

        self._last_kernel_cache_spec = None



    def _get_last_kernel(self, channels: int, device: torch.device):

        sigma = self.last_pool_sigma if self.last_pool_sigma > 0 else channels ** 0.5

        cache_spec = (channels, float(sigma), device.type, device.index)

        if self._last_kernel_cache is None or self._last_kernel_cache_spec != cache_spec:

            kernel = torch.exp(

                -0.5 * (torch.arange(-channels // 2 + 1, channels // 2 + 1, device=device, dtype=torch.float32) / sigma) ** 2

            )

            kernel = kernel / kernel.max().clamp_min(1e-12)

            self._last_kernel_cache = kernel.view(1, 1, -1)

            self._last_kernel_cache_spec = cache_spec

        return self._last_kernel_cache



    def _resolve_last_topk(self, num_patches: int):

        if self.last_pool_topk > 0:

            return min(self.last_pool_topk, num_patches)

        ratio = self.last_pool_topk_ratio if self.last_pool_topk_ratio > 0 else 1.0

        return min(max(int(round(num_patches * ratio)), 1), num_patches)



    def last_global_pool(self, patch_tokens: torch.Tensor):

        if patch_tokens.ndim != 3:

            raise ValueError(f"patch_tokens must be 3D, got shape={tuple(patch_tokens.shape)}")

        if patch_tokens.shape[1] <= 0:

            raise ValueError("patch_tokens must contain at least one patch token")



        work_tokens = patch_tokens.float()

        kernel = self._get_last_kernel(work_tokens.shape[-1], work_tokens.device)

        filtered = torch.fft.fft(work_tokens, dim=-1)

        filtered = torch.fft.fftshift(filtered, dim=-1)

        filtered = filtered * kernel

        filtered = torch.fft.ifftshift(filtered, dim=-1)

        filtered = torch.fft.ifft(filtered, dim=-1).real



        stability = work_tokens / (torch.abs(filtered - work_tokens) + self.last_pool_eps)

        topk = self._resolve_last_topk(work_tokens.shape[1])

        _, indices = torch.topk(stability, k=topk, dim=1, largest=True)

        selected = torch.gather(work_tokens, 1, indices)

        pooled = selected.mean(dim=1)

        return pooled.to(dtype=patch_tokens.dtype)

    



    def custom_attn(self, attn_layer, x, model_type='ClearCLIP'):



        num_heads = attn_layer.num_heads

        tgt_len, bsz, embed_dim = x.size()

        head_dim = embed_dim // num_heads

        scale = head_dim ** -0.5



        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)

        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)



        if model_type == 'vanilla':

            qk_attn = torch.bmm(q, k.transpose(1, 2)) * scale

            attn_weights = F.softmax(qk_attn, dim=-1)

        elif model_type == 'MaskCLIP':

            mask = torch.empty(q.shape[1], q.shape[1], dtype=q.dtype).to(q.device)

            mask.fill_(float('-inf'))

            mask.fill_diagonal_(0)

            mask = mask.unsqueeze(0).repeat(q.shape[0], 1, 1)

            attn_weights = F.softmax(mask, dim=-1)



        elif model_type == 'SCLIP':

            qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale

            kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale

            attn_weights = F.softmax(qq_attn, dim=-1) + F.softmax(kk_attn, dim=-1)



        elif model_type == 'ClearCLIP':

            qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale

            attn_weights = F.softmax(qq_attn, dim=-1)



        attn_output = torch.bmm(attn_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)

        attn_output = attn_layer.out_proj(attn_output)

        

        return attn_output



    def forward(self, x: torch.Tensor, model_type: str = 'ClearCLIP', ignore_residual=True, last_n_layers=1):

        H = x.size(2)

        W = x.size(3)

        if H == 224 and W == 224:

            self.positional_embedding_new = self.positional_embedding

        else:

            self.positional_embedding_new = self.upsample_pos_emb(self.positional_embedding, (H//16,W//16))

            

        x = self.conv1(x)                                  

        x = x.reshape(x.shape[0], x.shape[1], -1)                                 

        x = x.permute(0, 2, 1)                                 

        x = torch.cat(

            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),

             x], dim=1)                                     

        x = x + self.positional_embedding.to(x.dtype)



                                                                                                 

                                                               

        if self.VPT_shallow:

            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()

            x = torch.cat([x, visual_ctx], dim=1)

        else:

            assert self.prompt_till_layer_visual == 0



                               

        x = self.ln_pre(x)



        x = x.permute(1, 0, 2)              



        attn_weights = []

        for blk in self.transformer.resblocks[:-last_n_layers]:

            if self.need_weights:

                x, attn_w = blk(x)

                attn_weights.append(attn_w.detach())

            else:

                x = blk(x)

        output = 0

        for blk in self.transformer.resblocks[-last_n_layers:]:

            attn_output = self.custom_attn(blk.attn, blk.ln_1(x), model_type=model_type)

            if ignore_residual:

                output += attn_output

            else:

                x_out = x + attn_output

                x_out = x_out + blk.mlp(blk.ln_2(x_out))

                output += x_out

    

            if self.need_weights:

                x, attn_w = blk(x)

                attn_weights.append(attn_w.detach())

            else:

                x = blk(x)

            

        x_cur = output

        x_ori = x

        if self.VPT_shallow:

            x_cur = x_cur[:-self.n_ctx, :, :]

            x_ori = x_ori[:-self.n_ctx, :, :]      

            if self.need_weights:

                attn_weights = [aw[:, 1:-self.n_ctx, 1:-self.n_ctx] for aw in attn_weights]

        else:

            if self.need_weights:

                attn_weights = [aw[:, 1:, 1:] for aw in attn_weights]

        

        x_cur = x_cur.permute(1, 0, 2)              

        x_ori = x_ori.permute(1, 0, 2)

        x_ori_patch_tokens = x_ori[:, 1:, :]



        x_cur = self.ln_post(x_cur)

        cls_global = self.ln_post(x_ori[:, 0, :])

        last_global = None

        if self.last_pool_enabled and self.last_pool_mode != "off":

            last_global = self.ln_post(self.last_global_pool(x_ori_patch_tokens))



        if self.proj is not None:

            x_cur = x_cur @ self.proj

            cls_global = cls_global @ self.proj

            if last_global is not None:

                last_global = last_global @ self.proj



        self.last_cls_global_feature = cls_global

        self.last_last_global_feature = last_global



        if not self.last_pool_enabled or self.last_pool_mode == "off" or last_global is None:

            x_ori = cls_global

        elif self.last_pool_mode == "blend":

            alpha = min(max(self.last_pool_fusion_alpha, 0.0), 1.0)

            x_ori = (1.0 - alpha) * cls_global + alpha * last_global

        elif self.last_pool_mode == "consensus":

            x_ori = self._consensus_last_fusion(cls_global, last_global)

        elif self.last_pool_mode == "unseen_rescue":

            x_ori = cls_global

        else:

            x_ori = last_global



        return x_cur, x_ori, attn_weights





class CLIP(nn.Module):                                                    

    def __init__(self,

                 embed_dim: int,

                         

                 image_resolution: int,

                 vision_layers: Union[Tuple[int, int, int, int], int],

                 vision_width: int,

                 vision_patch_size: int,

                       

                 context_length: int,

                 vocab_size: int,

                 transformer_width: int,

                 transformer_heads: int,

                 transformer_layers: int,

                 design_details

                 ):

        super().__init__()



        self.context_length = context_length

        trainer = design_details['trainer']



        if isinstance(vision_layers, (tuple, list)):                                          

            vision_heads = vision_width * 32 // 64

            self.visual = ModifiedResNet(

                layers=vision_layers,

                output_dim=embed_dim,

                heads=vision_heads,

                input_resolution=image_resolution,

                width=vision_width

            )

        else:          

            vision_heads = vision_width // 64          

            self.visual = VisionTransformer(               

                input_resolution=image_resolution,

                patch_size=vision_patch_size,

                width=vision_width,

                layers=vision_layers,

                heads=vision_heads,

                output_dim=embed_dim,

                design_details=design_details                                                

            )

                                                                              

                                      

        prompt_till_layer_text = design_details['language_depth']                                              

        self.transformer = Transformer(                        

            width=transformer_width,

            layers=transformer_layers,

            heads=transformer_heads,

            attn_mask=self.build_attention_mask(),

            prompts_needed=prompt_till_layer_text,

            text_layer=True,

            need_weights=False,

            design_details=design_details

        )



        self.vocab_size = vocab_size         

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)                                                          

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))            

        self.ln_final = LayerNorm(transformer_width)                                                 



        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))                                                                             

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))                                        



        self.initialize_parameters()                                            



    def initialize_parameters(self):                              

        nn.init.normal_(self.token_embedding.weight, std=0.02)                   

        nn.init.normal_(self.positional_embedding, std=0.01)



        if isinstance(self.visual, ModifiedResNet):                   

            if self.visual.attnpool is not None:                 

                std = self.visual.attnpool.c_proj.in_features ** -0.5

                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)

                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)

                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)

                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)



            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:

                for name, param in resnet_block.named_parameters():

                    if name.endswith("bn3.weight"):                                  

                        nn.init.zeros_(param)



        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)                      

        attn_std = self.transformer.width ** -0.5

        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:

            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)

            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)

            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)

            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)



        if self.text_projection is not None:

            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)



    def build_attention_mask(self):  

                                                                                            

                                                              

        mask = torch.empty(self.context_length, self.context_length)

        mask.fill_(float("-inf"))

        mask.triu_(1)                               

        return mask



    @property

    def dtype(self):

        return self.visual.conv1.weight.dtype



    def encode_image(self, image):

        return self.visual(image.type(self.dtype))



    def encode_text(self, text):

        x = self.token_embedding(text).type(self.dtype)                                



        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)              

        x = self.transformer(x)

        x = x.permute(1, 0, 2)              

        x = self.ln_final(x).type(self.dtype)



                                                                                                 

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection



        return x



    def forward(self, image, text):

        image_features = self.encode_image(image)

        text_features = self.encode_text(text)



                             

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



                                     

        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_text = logit_scale * text_features @ image_features.t()



        return logits_per_image, logits_per_text





def convert_weights(model: nn.Module):                       

                                                     



    def _convert_weights_to_fp16(l):                                             

        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):

            l.weight.data = l.weight.data.half()

            if l.bias is not None:

                l.bias.data = l.bias.data.half()



        if isinstance(l, nn.MultiheadAttention):                                                   

            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:

                tensor = getattr(l, attr)

                if tensor is not None:

                    tensor.data = tensor.data.half()



        for name in ["text_projection", "proj"]:       

            if hasattr(l, name):

                attr = getattr(l, name)

                if attr is not None:

                    attr.data = attr.data.half()



    model.apply(_convert_weights_to_fp16)              





def build_model(state_dict: dict, design_details):                                     

    vit = "visual.proj" in state_dict                                          



    if vit:

        vision_width = state_dict["visual.conv1.weight"].shape[0]                          

        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])                                                                                        

        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]                                             

        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)                                      

        image_resolution = vision_patch_size * grid_size          

    else:                                         

        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]

        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]

        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)

        vision_patch_size = None

        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]

        image_resolution = output_width * 32



    embed_dim = state_dict["text_projection"].shape[1]                        

    context_length = state_dict["positional_embedding"].shape[0]                       

    vocab_size = state_dict["token_embedding.weight"].shape[0]        

    transformer_width = state_dict["ln_final.weight"].shape[0]                         

    transformer_heads = transformer_width // 64                                                      

    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))                                



    model = CLIP(                                   

        embed_dim,

        image_resolution, vision_layers, vision_width, vision_patch_size,

        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details

    )



    for key in ["input_resolution", "context_length", "vocab_size"]:                                                           

        if key in state_dict:

            del state_dict[key]



    convert_weights(model)

    try:

        model.load_state_dict(state_dict)                                         

    except:

        missing_keys, _ = model.load_state_dict(state_dict, strict=False)

        print('Weights not found for some missing keys: ', missing_keys)

    return model.eval()                                   

