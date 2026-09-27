import argparse

import csv

import datetime

import hashlib

import json

import os

import platform

import shlex

import subprocess
import sys
from pathlib import Path

import numpy as np

import torch

import torch.backends.cudnn as cudnn

import torch.nn as nn

import torch.nn.functional as F

from torch.cuda.amp import autocast

from torch.utils.data import DataLoader

from tqdm import tqdm



from datasets import build_dataset

from model import build_text_model

from utils.build_cfg import setup_cfg

from utils.helper import (

    compute_AP_openimages_biam,

    compute_F1,

    compute_weighted_map_percent,

    filter_openimages_biam_classes,

    init_log,

    is_openimagesv4_dataset,

    setup_seed,

    write_description_to_folder,

)





parser = argparse.ArgumentParser(description="Evaluate an existing checkpoint without retraining")

parser.add_argument("--config_file", type=str, default=None, help="model config file path; defaults to checkpoint sibling configs.txt")

parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path to evaluate")

parser.add_argument("--output_dir", type=str, default=None, help="directory to save eval logs/results")

parser.add_argument("--use_dataparallel", action="store_true", help="wrap model with DataParallel when multiple GPUs are visible")

parser.add_argument(

    "opts",

    default=None,

    nargs=argparse.REMAINDER,

    help="Override config options using KEY VALUE pairs, e.g. CLASS_RERANK.PATCH_TOPK 6",

)





def get_prompt_learner(model):

    if isinstance(model, nn.DataParallel):

        return model.module.prompt_learner

    return model.prompt_learner





def get_model_module(model):

    if isinstance(model, nn.DataParallel):

        return model.module

    return model





def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")





def safe_torch_load(checkpoint_path):

                                                                              

                                                                                             

    try:

        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    except TypeError:

        return torch.load(checkpoint_path, map_location="cpu")





def resolve_config_file(args):         

    if args.config_file:

        config_path = Path(args.config_file)

        if not config_path.is_file():

            raise FileNotFoundError(f"Config file not found: {config_path}")

        return str(config_path)



    checkpoint_path = Path(args.checkpoint)

    candidate_paths = [

        checkpoint_path.parent / "configs.txt",

        checkpoint_path.parent / "config.yaml",

        checkpoint_path.parent / "config.yml",

    ]

    for candidate in candidate_paths:

        if candidate.is_file():

            return str(candidate)



    raise FileNotFoundError(

        "No config_file was provided and no sibling configs.txt/config.yaml/config.yml "

        f"was found next to checkpoint: {checkpoint_path}"

    )





def normalize_prompt_state_dict(state_dict):

    normalized = {}

    for key, value in state_dict.items():

        if key.startswith("module.prompt_learner."):

            key = key[len("module.prompt_learner."):]

        elif key.startswith("prompt_learner."):

            key = key[len("prompt_learner."):]

        elif key.startswith("module."):

            key = key[len("module."):]

        normalized[key] = value

    return normalized



                                               

def extract_prompt_state(checkpoint_obj):                                                       

    meta = {}                                                

    

    if isinstance(checkpoint_obj, dict):                                     

        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):

            meta["checkpoint_type"] = "full_checkpoint"

            meta["checkpoint_epoch"] = checkpoint_obj.get("epoch")

            meta["best_unseen_F1"] = checkpoint_obj.get("best_unseen_F1")

            meta["best_unseen_mAP"] = checkpoint_obj.get("best_unseen_mAP")

            meta["best_gzsl_F1"] = checkpoint_obj.get("best_gzsl_F1")

            meta["best_unseen_epoch"] = checkpoint_obj.get("best_unseen_epoch")

            meta["best_unseen_mAP_epoch"] = checkpoint_obj.get("best_unseen_mAP_epoch")

            meta["best_gzsl_epoch"] = checkpoint_obj.get("best_gzsl_epoch")

            meta["best_monitor_name"] = checkpoint_obj.get("best_monitor_name")

            meta["best_monitor_value"] = checkpoint_obj.get("best_monitor_value")

            meta["best_monitor_epoch"] = checkpoint_obj.get("best_monitor_epoch")

            return normalize_prompt_state_dict(checkpoint_obj["state_dict"]), meta                                                                                  



        if "prompt_learner" in checkpoint_obj and isinstance(checkpoint_obj["prompt_learner"], dict):                                  

            meta["checkpoint_type"] = "prompt_learner_dict"

            return normalize_prompt_state_dict(checkpoint_obj["prompt_learner"]), meta                          



        prompt_like_keys = {"clsn", "token_prefix", "token_suffix", "tokenized_prompt"}

        if prompt_like_keys & set(checkpoint_obj.keys()):                               

            meta["checkpoint_type"] = "prompt_state_dict"

            return normalize_prompt_state_dict(checkpoint_obj), meta



    raise ValueError(

        "Unsupported checkpoint format. Expected a full checkpoint with 'state_dict' "

        "or a prompt-learner state dict containing keys like 'clsn'."

    )





def compute_ap_with_details(predictions, labels):

    num_class = predictions.size(1)

    ap = torch.zeros(num_class, dtype=torch.float32, device=predictions.device)          

    rows = []



    for idx_cls in range(num_class):

        prediction = predictions[:, idx_cls]

        label = labels[:, idx_cls]

        ignored_count = 0



        if (label == -1).sum() != 0:

            mask = label.abs() == 1

            ignored_count = int((~mask).sum().item())

            label = torch.clamp(label[mask], min=0, max=1)

            prediction = prediction[mask]



        positive_count = int((label > 0).sum().item())

        valid_count = int(label.numel())

        negative_count = valid_count - positive_count

        ap_value = 0.0



        if positive_count > 0:

            _, sort_idx = prediction.sort(descending=True)

            sorted_label = label[sort_idx]

            positive_mask = (sorted_label == 1).float()

            tp = positive_mask.cumsum(0)

            fp = (sorted_label != 1).float().cumsum(0)

            precision = tp / (tp + fp)

            ap_value = float((positive_mask * precision).sum().item() / positive_count)

            ap[idx_cls] = ap_value



        rows.append(

            {

                "class_index": idx_cls,

                "ap": ap_value,

                "ap_percent": 100.0 * ap_value,

                "positive_count": positive_count,

                "negative_count": negative_count,

                "ignored_count": ignored_count,

                "valid_count": valid_count,

                "positive_ratio": float(positive_count / valid_count) if valid_count > 0 else 0.0,

            }

        )



    return ap, rows





def compute_ap_with_details_openimages_biam(predictions, labels):

    ap, positive_counts, negative_counts, ignored_counts, valid_counts = compute_AP_openimages_biam(

        predictions, labels, return_stats=True

    )

    rows = []

    for idx_cls in range(predictions.size(1)):

        positive_count = int(positive_counts[idx_cls].item())

        negative_count = int(negative_counts[idx_cls].item())

        ignored_count = int(ignored_counts[idx_cls].item())

        valid_count = int(valid_counts[idx_cls].item())

        ap_value = float(ap[idx_cls].item())

        rows.append(

            {

                "class_index": idx_cls,

                "ap": ap_value,

                "ap_percent": 100.0 * ap_value,

                "positive_count": positive_count,

                "negative_count": negative_count,

                "ignored_count": ignored_count,

                "valid_count": valid_count,

                "positive_ratio": float(positive_count / valid_count) if valid_count > 0 else 0.0,

            }

        )

    return ap, rows, positive_counts





def _filter_class_names_by_mask(class_names, class_mask):

    if class_names is None:

        return None

    indices = torch.nonzero(class_mask, as_tuple=False).flatten().tolist()

    return [class_names[idx] for idx in indices]





def save_per_class_ap_report(report_dir, prefix, class_names, rows, weighted_map=None):

    report_dir = Path(report_dir)

    if len(class_names) != len(rows):

        raise ValueError(

            f"Class name count mismatch for {prefix}: expected {len(rows)}, got {len(class_names)}"

        )



    ranked_indices = sorted(range(len(rows)), key=lambda idx: rows[idx]["ap"], reverse=True)

    rank_map = {row_idx: rank + 1 for rank, row_idx in enumerate(ranked_indices)}

    report_rows = []

    for idx, row in enumerate(rows):

        report_row = dict(row)

        report_row["class_name"] = str(class_names[idx])

        report_row["ap_rank_desc"] = int(rank_map[idx])

        report_rows.append(report_row)



    csv_path = report_dir / f"per_class_ap_{prefix}.csv"

    json_path = report_dir / f"per_class_ap_{prefix}.json"

    fieldnames = [

        "class_index",

        "class_name",

        "ap",

        "ap_percent",

        "ap_rank_desc",

        "positive_count",

        "negative_count",

        "ignored_count",

        "valid_count",

        "positive_ratio",

    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as fp:

        writer = csv.DictWriter(fp, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(report_rows)



    summary_payload = {

        "prefix": prefix,

        "num_classes": len(report_rows),

        "mAP": float(np.mean([row["ap_percent"] for row in report_rows])),

        "rows": report_rows,

    }

    if weighted_map is not None:

        summary_payload["weighted_mAP"] = float(weighted_map)

    with open(json_path, "w", encoding="utf-8") as fp:

        json.dump(summary_payload, fp, indent=2, ensure_ascii=False)





def get_eval_topk_list(dataset_name):

    if is_openimagesv4_dataset(dataset_name):

        return [10, 20]

    return [3, 5]





def compute_metric_block(predictions, labels, prefix, class_names=None, report_dir=None, dataset_name=None):

    weighted_map = None

    report_class_names = class_names



    if is_openimagesv4_dataset(dataset_name):

        eval_predictions, eval_labels, class_mask = filter_openimages_biam_classes(predictions, labels, prefix)

        ap, per_class_rows, positive_counts = compute_ap_with_details_openimages_biam(eval_predictions, eval_labels)

        mAP = 100 * ap.mean().item()

        weighted_map = compute_weighted_map_percent(ap, positive_counts)

        kept_indices = torch.nonzero(class_mask, as_tuple=False).flatten().tolist()

        for row, original_idx in zip(per_class_rows, kept_indices):

            row["class_index"] = int(original_idx)

        if class_names is not None:

            report_class_names = _filter_class_names_by_mask(class_names, class_mask)

    else:

        ap, per_class_rows = compute_ap_with_details(predictions, labels)

        mAP = 100 * ap.mean().item()



    if report_class_names is not None and report_dir is not None:

        save_per_class_ap_report(

            report_dir,

            prefix,

            report_class_names,

            per_class_rows,

            weighted_map=weighted_map,

        )



    topk_list = get_eval_topk_list(dataset_name)

    metrics = {f"mAP_{prefix}": float(mAP)}

    for k_val in topk_list:

        preds_topk = predictions.clone()

        f1_k, p_k, r_k = compute_F1(preds_topk, labels, k_val=k_val)

        metrics.update(

            {

                f"P_{k_val}_{prefix}": float(p_k.item()),

                f"R_{k_val}_{prefix}": float(r_k.item()),

                f"F1_{k_val}_{prefix}": float(f1_k.item()),

            }

        )

    if weighted_map is not None:

        metrics[f"weighted_mAP_{prefix}"] = float(weighted_map)

    return metrics





def to_serializable(value):

    if torch.is_tensor(value):

        if value.numel() == 1:

            return value.item()

        return value.tolist()

    return value





def _run_metadata_command(command):

    try:

        return subprocess.check_output(

            command,

            cwd=Path(__file__).resolve().parent,

            stderr=subprocess.STDOUT,

            text=True,

            timeout=15,

        ).strip()

    except Exception as exc:

        return f"unavailable: {type(exc).__name__}: {exc}"





def _sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as fp:

        for block in iter(lambda: fp.read(1024 * 1024), b""):

            digest.update(block)

    return digest.hexdigest()





def save_reproducibility_metadata(record_path, checkpoint_path, config_path):

    command = shlex.join([sys.executable, *sys.argv])

    (record_path / "command.txt").write_text(command + "\n", encoding="utf-8")



    environment = {

        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),

        "hostname": platform.node(),

        "platform": platform.platform(),

        "python_executable": sys.executable,

        "python_version": sys.version,

        "torch_version": torch.__version__,

        "cuda_runtime": torch.version.cuda,

        "cuda_available": bool(torch.cuda.is_available()),

        "cudnn_version": torch.backends.cudnn.version(),

        "gpu_count": int(torch.cuda.device_count()),

        "gpus": [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())],

        "nvidia_smi": _run_metadata_command(

            [

                "nvidia-smi",

                "--query-gpu=index,name,driver_version,memory.total",

                "--format=csv,noheader",

            ]

        ),

        "git_commit": _run_metadata_command(["git", "rev-parse", "HEAD"]),

        "git_status_short": _run_metadata_command(["git", "status", "--short", "--untracked-files=no"]),

        "working_directory": str(Path.cwd()),

        "config_path": str(Path(config_path).resolve()),

        "checkpoint_path": str(Path(checkpoint_path).resolve()),

        "checkpoint_sha256": _sha256_file(checkpoint_path),

        "command": command,

    }

    with open(record_path / "environment.json", "w", encoding="utf-8") as fp:

        json.dump(environment, fp, indent=2, ensure_ascii=False)

    return environment





def format_metrics(metrics, dataset_name=None):

    topk_list = get_eval_topk_list(dataset_name)

    unseen_parts = [f"mAP={metrics['mAP_unseen']:.4f}"]

    glob_parts = [f"mAP={metrics['mAP_glob']:.4f}"]

    for k_val in topk_list:

        unseen_parts.append(

            "P_{k}={p:.4f}, R_{k}={r:.4f}, F1_{k}={f1:.4f}".format(

                k=k_val,

                p=metrics[f"P_{k_val}_unseen"],

                r=metrics[f"R_{k_val}_unseen"],

                f1=metrics[f"F1_{k_val}_unseen"],

            )

        )

        glob_parts.append(

            "P_{k}={p:.4f}, R_{k}={r:.4f}, F1_{k}={f1:.4f}".format(

                k=k_val,

                p=metrics[f"P_{k_val}_glob"],

                r=metrics[f"R_{k_val}_glob"],

                f1=metrics[f"F1_{k_val}_glob"],

            )

        )

    text = "unseen: " + "\n, ".join(unseen_parts) + "\n" + "glob:   " + "\n, ".join(glob_parts)

    if "weighted_mAP_unseen" in metrics or "weighted_mAP_glob" in metrics:

        weighted_lines = []

        if "weighted_mAP_unseen" in metrics:

            weighted_lines.append(f"weighted_mAP_unseen={metrics['weighted_mAP_unseen']:.4f}")

        if "weighted_mAP_glob" in metrics:

            weighted_lines.append(f"weighted_mAP_glob={metrics['weighted_mAP_glob']:.4f}")

        text += "\n" + " ".join(weighted_lines)

    return text





def get_eval_tta_info(model_module, cfg):

    if hasattr(model_module, "get_eval_tta_info"):

        return model_module.get_eval_tta_info()

    return {

        "enabled": bool(getattr(cfg.TTA, "ENABLED", False)),

        "mode": "disabled",

        "scope": "all",

        "include_base_prompt": False,

        "aggregation": "none",

        "base_weight": 1.0,

        "template_weights": [],

        "num_templates": 0,

        "templates": [],

    }





def get_eval_chunk_info(model_module, cfg):

    if hasattr(model_module, "get_eval_chunk_info"):

        return model_module.get_eval_chunk_info()

    return {

        "enabled": bool(getattr(cfg.CHUNK, "ENABLED", False)),

        "mode": "disabled",

        "scope": "all",

        "size": 0,

        "topk_per_chunk": 0,

        "filter_value": -1e4,

        "balanced": True,

    }





def get_eval_rerank_info(model_module, cfg):

    if hasattr(model_module, "get_eval_rerank_info"):

        return model_module.get_eval_rerank_info()

    return {

        "enabled": bool(getattr(cfg.RERANK, "ENABLED", False)),

        "mode": "disabled",

        "scope": "all",

        "topk": 0,

        "sim_temperature": 0.2,

        "logit_temperature": 1.0,

    }





def get_eval_class_rerank_info(cfg):

    return {

        "enabled": bool(getattr(cfg.CLASS_RERANK, "ENABLED", False)),

        "scope": str(getattr(cfg.CLASS_RERANK, "SCOPE", "unseen_only")),

        "visual_topk": int(getattr(cfg.CLASS_RERANK, "VISUAL_TOPK", 20)),

        "visual_chunk_size": int(getattr(cfg.CLASS_RERANK, "VISUAL_CHUNK_SIZE", 256)),

        "visual_temperature": float(getattr(cfg.CLASS_RERANK, "VISUAL_TEMPERATURE", 0.10)),

        "patch_topk": int(getattr(cfg.CLASS_RERANK, "PATCH_TOPK", 4)),

        "patch_weight": float(getattr(cfg.CLASS_RERANK, "PATCH_WEIGHT", 0.10)),

        "margin_weight": float(getattr(cfg.CLASS_RERANK, "MARGIN_WEIGHT", 0.08)),

        "knn_weight": float(getattr(cfg.CLASS_RERANK, "KNN_WEIGHT", 0.18)),

        "relative_knn_enabled": bool(getattr(cfg.CLASS_RERANK, "RELATIVE_KNN_ENABLED", False)),

        "relative_knn_topk": int(getattr(cfg.CLASS_RERANK, "RELATIVE_KNN_TOPK", 3)),

        "relative_knn_weight": float(getattr(cfg.CLASS_RERANK, "RELATIVE_KNN_WEIGHT", 0.05)),

        "patch_local_blend": float(getattr(cfg.CLASS_RERANK, "PATCH_LOCAL_BLEND", 0.5)),

        "zscore_clamp": float(getattr(cfg.CLASS_RERANK, "ZSCORE_CLAMP", 5.0)),

        "knn_score_chunk_size": int(getattr(cfg.CLASS_RERANK, "KNN_SCORE_CHUNK_SIZE", 2048)),

    }





def log_eval_tta_info(logger, info):

    if not info["enabled"]:

        return

    logger.info(

        "Eval TTA: enabled=%s mode=%s scope=%s include_base=%s aggregation=%s templates=%d",

        info["enabled"],

        info["mode"],

        info["scope"],

        info["include_base_prompt"],

        info["aggregation"],

        info["num_templates"],

    )

    if info["aggregation"] == "weighted_feature_mean":

        logger.info(

            "Eval TTA weights: base_weight=%.4f template_weights=%s",

            float(info["base_weight"]),

            info["template_weights"],

        )





def log_eval_chunk_info(logger, info):

    if not info["enabled"]:

        return

    logger.info(

        "Eval Chunk: enabled=%s mode=%s scope=%s size=%d topk_per_chunk=%d balanced=%s filter_value=%.1f",

        info["enabled"],

        info["mode"],

        info["scope"],

        int(info["size"]),

        int(info["topk_per_chunk"]),

        info["balanced"],

        float(info["filter_value"]),

    )





def log_eval_rerank_info(logger, info):

    if not info["enabled"]:

        return

    logger.info(

        "Eval Rerank: enabled=%s mode=%s scope=%s topk=%d sim_temp=%.3f logit_temp=%.3f gate_mode=%s gate_topk=%d gate_threshold=%.3f",

        info["enabled"],

        info["mode"],

        info["scope"],

        int(info["topk"]),

        float(info["sim_temperature"]),

        float(info["logit_temperature"]),

        info.get("gate_mode", "none"),

        int(info.get("gate_topk", 0)),

        float(info.get("gate_threshold", 0.0)),

    )





def log_eval_class_rerank_info(logger, info):

    if not info["enabled"]:

        return

    logger.info(

        "Eval ClassRerank: enabled=%s scope=%s visual_topk=%d visual_temp=%.3f patch_topk=%d "

        "patch_weight=%.3f margin_weight=%.3f knn_weight=%.3f relative_knn_enabled=%s "

        "relative_knn_topk=%d relative_knn_weight=%.3f patch_local_blend=%.3f zscore_clamp=%.2f",

        info["enabled"],

        info["scope"],

        int(info["visual_topk"]),

        float(info["visual_temperature"]),

        int(info["patch_topk"]),

        float(info["patch_weight"]),

        float(info["margin_weight"]),

        float(info["knn_weight"]),

        bool(info.get("relative_knn_enabled", False)),

        int(info.get("relative_knn_topk", 0)),

        float(info.get("relative_knn_weight", 0.0)),

        float(info["patch_local_blend"]),

        float(info["zscore_clamp"]),

    )





def _safe_classwise_zscore(values, clamp_value):              

    values = values.float()                                                              

    mean = values.mean(dim=0, keepdim=True)

    std = values.std(dim=0, unbiased=False, keepdim=True)

    zscore = (values - mean) / std.clamp_min(1e-6)

    if clamp_value > 0:

        zscore = torch.clamp(zscore, min=-clamp_value, max=clamp_value)

    return zscore





def _compute_visual_knn_scores(base_scores, visual_features, info):

    num_images = visual_features.shape[0]

    if num_images <= 1:

        return base_scores.clone()



    requested_topk = int(info["visual_topk"])

    if requested_topk < 0:

        raise ValueError(f"CLASS_RERANK.VISUAL_TOPK must be non-negative, got {requested_topk}")

    if requested_topk == 0:

                                                                           

                                                                            

                                                                            

        return base_scores.clone()



    topk = min(requested_topk, num_images - 1)              

    chunk_size = max(1, int(info["visual_chunk_size"]))        

    temperature = float(info["visual_temperature"])                  

    if temperature <= 0:

        raise ValueError(f"CLASS_RERANK.VISUAL_TEMPERATURE must be positive, got {temperature}")



    feature_dtype = torch.float16 if visual_features.device.type == "cuda" else torch.float32

    visual_features = F.normalize(visual_features.to(dtype=feature_dtype), dim=1)           

    device = visual_features.device

    neighbors = torch.empty((num_images, topk), dtype=torch.long, device=device)

    weights = torch.empty((num_images, topk), dtype=torch.float32, device=device)



    for start in range(0, num_images, chunk_size):                                                      

        end = min(start + chunk_size, num_images)

        sim = visual_features[start:end] @ visual_features.t()        

        row_idx = torch.arange(start, end, device=device)

        sim[torch.arange(end - start, device=device), row_idx] = -1e4

        sim_topk, idx_topk = torch.topk(sim, k=topk, dim=1)

        neighbors[start:end] = idx_topk                 

        weights[start:end] = F.softmax(sim_topk / temperature, dim=1).to(dtype=torch.float32)                 



    score_chunk_size = max(1, int(info["knn_score_chunk_size"]))                    

    knn_scores = torch.empty_like(base_scores, dtype=torch.float32)

    for start in range(0, base_scores.shape[1], score_chunk_size):

        end = min(start + score_chunk_size, base_scores.shape[1])

        neighbor_scores = base_scores[neighbors, start:end]

        knn_scores[:, start:end] = (neighbor_scores * weights.unsqueeze(-1)).sum(dim=1)



    return knn_scores                                                                     





def _build_relative_knn_confuser_index(text_features, info, num_seen_classes):

    if not bool(info.get("relative_knn_enabled", False)):                     

        return None



    scope = str(info["scope"]).lower()

    if scope == "glob":

        scope = "all"



    total_classes = text_features.shape[0]

    if scope == "all":

        target_offset = 0

        target_count = total_classes

    elif scope == "unseen_only":        

        if num_seen_classes is None:

            raise ValueError("Relative-KNN requires num_seen_classes when CLASS_RERANK.SCOPE='unseen_only'")

        target_offset = int(num_seen_classes)                       

        target_count = total_classes - target_offset                                                                  

    else:

        raise ValueError(f"Unsupported CLASS_RERANK.SCOPE: {info['scope']}")



    if target_count <= 1:

        return None



    confuser_topk = min(max(int(info.get("relative_knn_topk", 0)), 0), target_count - 1)

    if confuser_topk <= 0:

        return None



    target_text = F.normalize(  

        text_features[target_offset : target_offset + target_count].float(),

        dim=1,

    )

    proto_sim = target_text @ target_text.t()                                                                            

    proto_sim.fill_diagonal_(-1e4)

    return torch.topk(proto_sim, k=confuser_topk, dim=1).indices.cpu()                             





def _compute_relative_knn_feature(knn_scores, relative_knn_confuser_idx):                                                                

    if relative_knn_confuser_idx is None or relative_knn_confuser_idx.numel() == 0:

        return torch.zeros_like(knn_scores)

                                                                                                       

    target_count = knn_scores.shape[1]

    confuser_idx = relative_knn_confuser_idx.to(device=knn_scores.device, dtype=torch.long)

    expanded_scores = knn_scores.unsqueeze(1).expand(-1, target_count, -1)

    gather_idx = confuser_idx.unsqueeze(0).expand(knn_scores.shape[0], -1, -1)

    confuser_knn_scores = torch.gather(expanded_scores, dim=2, index=gather_idx)

    strongest_confuser = confuser_knn_scores.max(dim=2).values

    return knn_scores - strongest_confuser                                                    

                                                              



def apply_eval_classwise_rerank(              

    preds_glob, 

    patch_feature_all,                                      

    visual_features,

    info,

    num_seen_classes,

    relative_knn_confuser_idx=None,

):                                                  

    if not info["enabled"]:    

        return preds_glob[:, num_seen_classes:].clone(), preds_glob

    scope = str(info["scope"]).lower()

    if scope == "glob":                       

        scope = "all"

    if scope not in {"unseen_only", "all"}:

        raise ValueError(f"Unsupported CLASS_RERANK.SCOPE: {info['scope']}")

    if preds_glob.numel() == 0: 

        return preds_glob[:, num_seen_classes:].clone(), preds_glob



    total_classes = preds_glob.shape[1]

    if scope == "all":     

        target_offset = 0

        target_count = total_classes

    else:

        target_offset = int(num_seen_classes)                       

        target_count = total_classes - target_offset                      

    if target_count <= 0:      

        return preds_glob[:, num_seen_classes:].clone(), preds_glob



    device = torch.device("cuda") if torch.cuda.is_available() else preds_glob.device                                                                 

    preds_glob = preds_glob.to(device=device, dtype=torch.float32)

    patch_feature_all = patch_feature_all.to(device=device, dtype=torch.float32)

    visual_features = visual_features.to(device=device, dtype=torch.float32)



    target_scores = preds_glob[:, target_offset : target_offset + target_count]                                         

    patch_feature = patch_feature_all[:, target_offset : target_offset + target_count]                                    

    zscore_clamp = float(info["zscore_clamp"])



    if total_classes <= 1:

        confuser = torch.zeros_like(target_scores)

    else:

        top2 = torch.topk(preds_glob, k=min(2, total_classes), dim=1).values                               

        top1 = top2[:, :1]

        second = top2[:, 1:2] if top2.shape[1] > 1 else top1

        confuser_all = torch.where(preds_glob == top1, second, top1).expand_as(preds_glob)                                                                                            

        confuser = confuser_all[:, target_offset : target_offset + target_count]                                     

    margin_feature = target_scores - confuser                                                              



    knn_scores = _compute_visual_knn_scores(target_scores, visual_features, info)                                                                               

    knn_feature = knn_scores - target_scores                                                              

    relative_knn_feature = _compute_relative_knn_feature(knn_scores, relative_knn_confuser_idx)            



    patch_term = _safe_classwise_zscore(patch_feature, zscore_clamp) * float(info["patch_weight"])                                                  

    margin_term = _safe_classwise_zscore(margin_feature, zscore_clamp) * float(info["margin_weight"])   

    knn_term = _safe_classwise_zscore(knn_feature, zscore_clamp) * float(info["knn_weight"])

    if bool(info.get("relative_knn_enabled", False)):

        relative_knn_term = (

            _safe_classwise_zscore(relative_knn_feature, zscore_clamp)

            * float(info.get("relative_knn_weight", 0.0))

        )

    else:

        relative_knn_term = torch.zeros_like(target_scores)



    reranked_target = target_scores + patch_term + margin_term + knn_term + relative_knn_term                                                                                              

    reranked_glob = preds_glob.clone()

    reranked_glob[:, target_offset : target_offset + target_count] = reranked_target

    reranked_unseen = reranked_glob[:, num_seen_classes:]

    return reranked_unseen.cpu(), reranked_glob.cpu()





def evaluate_standard(

    data_loader,

    model,

    logger,

    num_seen_classes=None,

    report_dir=None,

    unseen_class_names=None,

    all_class_names=None,

):

    model.eval()

    preds_unseen = []

    preds_glob = []

    labels_unseen = []

    labels_glob = []

    model_module = get_model_module(model)

    tta_info = get_eval_tta_info(model_module, model_module.cfg)

    log_eval_tta_info(logger, tta_info)

    rerank_info = get_eval_rerank_info(model_module, model_module.cfg)

    log_eval_rerank_info(logger, rerank_info)

    class_rerank_info = get_eval_class_rerank_info(model_module.cfg)

    log_eval_class_rerank_info(logger, class_rerank_info)

    chunk_info = get_eval_chunk_info(model_module, model_module.cfg)

    log_eval_chunk_info(logger, chunk_info)

    text_feats = None

    effective_num_seen_classes = num_seen_classes

    relative_knn_confuser_idx = None

    patch_features = []

    visual_feats = []



    with torch.no_grad():

        if not (tta_info["enabled"] and tta_info["scope"] == "unseen_only" and effective_num_seen_classes is None):

            with autocast():

                text_feats = model_module.encode_text_features(

                    use_tta=tta_info["enabled"],

                    num_seen_classes=effective_num_seen_classes,

                )

            if class_rerank_info["enabled"] and effective_num_seen_classes is not None:

                relative_knn_confuser_idx = _build_relative_knn_confuser_index(

                    text_feats,

                    class_rerank_info,

                    effective_num_seen_classes,

                )



    for images, seen_labels, unseen_labels in tqdm(data_loader):

        device = get_device()

        images = images.to(device)

        num_seen_cls = seen_labels.size(1)

        if effective_num_seen_classes is None:

            effective_num_seen_classes = int(num_seen_cls)

        if text_feats is None:

            with torch.no_grad():

                with autocast():

                    text_feats = model_module.encode_text_features(

                        use_tta=tta_info["enabled"],

                        num_seen_classes=num_seen_cls,

                    )

        elif effective_num_seen_classes is not None and num_seen_cls != effective_num_seen_classes:

            raise ValueError(

                f"Inconsistent num_seen_classes between dataset ({effective_num_seen_classes}) and batch ({num_seen_cls})"

            )

        if class_rerank_info["enabled"] and relative_knn_confuser_idx is None:

            relative_knn_confuser_idx = _build_relative_knn_confuser_index(

                text_feats,

                class_rerank_info,

                effective_num_seen_classes,

            )



        with torch.no_grad():

            with autocast():

                output_bundle = model_module.forward_for_open(

                    images,

                    text_feats,

                    num_seen_classes=num_seen_cls,

                    return_rerank_bundle=class_rerank_info["enabled"],

                )

                output = output_bundle["logit_final"] if class_rerank_info["enabled"] else output_bundle

                output = model_module.apply_eval_proto_rerank(

                    output, text_feats, num_seen_classes=num_seen_cls

                )

                output = model_module.apply_eval_label_chunking(output, num_seen_classes=num_seen_cls)

            logits_unseen = output[:, num_seen_cls:]

            logits_glob = output



        preds_unseen.append(logits_unseen.float().cpu())

        preds_glob.append(logits_glob.float().cpu())

        labels_unseen.append(unseen_labels)

        labels_glob.append(torch.cat([seen_labels, unseen_labels], dim=1))

        if class_rerank_info["enabled"]:

            patch_topk = max(1, min(int(class_rerank_info["patch_topk"]), output_bundle["logit_local"].shape[2]))

            local_topk = torch.topk(output_bundle["logit_local"], k=patch_topk, dim=2).values.mean(dim=2)

            local_mean = output_bundle["logit_local"].mean(dim=2)

            local_gap = local_mean - output_bundle["logit_glob"]

            patch_feature = (

                float(class_rerank_info["patch_local_blend"]) * (local_topk - local_mean)

                + (1.0 - float(class_rerank_info["patch_local_blend"])) * local_gap

            )

            patch_features.append(patch_feature.float().cpu())

            visual_feats.append(output_bundle["visual_feature"].float().cpu())



    preds_unseen = torch.cat(preds_unseen, dim=0)

    preds_glob = torch.cat(preds_glob, dim=0)

    labels_unseen = torch.cat(labels_unseen, dim=0)

    labels_glob = torch.cat(labels_glob, dim=0)

    if class_rerank_info["enabled"]:

        if torch.cuda.is_available():

            model.to(torch.device("cpu"))

            torch.cuda.empty_cache()

        preds_unseen, preds_glob = apply_eval_classwise_rerank(

            preds_glob,

            torch.cat(patch_features, dim=0),

            torch.cat(visual_feats, dim=0),

            class_rerank_info,

            num_seen_classes=effective_num_seen_classes,

            relative_knn_confuser_idx=relative_knn_confuser_idx,

        )



    unseen_mask = (labels_unseen > 0).sum(1) > 0

    unseen_metrics = compute_metric_block(

        preds_unseen[unseen_mask],

        labels_unseen[unseen_mask],

        "unseen",

        class_names=unseen_class_names,

        report_dir=report_dir,

        dataset_name=model_module.cfg.DATASET.NAME,

    )



    glob_mask = (labels_glob > 0).sum(1) > 0

    glob_metrics = compute_metric_block(

        preds_glob[glob_mask],

        labels_glob[glob_mask],

        "glob",

        class_names=all_class_names,

        report_dir=report_dir,

        dataset_name=model_module.cfg.DATASET.NAME,

    )



    metrics = {**unseen_metrics, **glob_metrics}

    topk_list = get_eval_topk_list(model_module.cfg.DATASET.NAME)

    logger.info("Evaluating predictions over all images")

    unseen_line = [f"mAP_unseen {metrics['mAP_unseen']:.2f}"]

    glob_line = [f"mAP_glob {metrics['mAP_glob']:.2f}"]

    for k_val in topk_list:

        unseen_line.append(

            "P_{k}_unseen {p:.4f} | R_{k}_unseen {r:.4f} | F1_{k}_unseen {f1:.4f}".format(

                k=k_val,

                p=metrics[f"P_{k_val}_unseen"],

                r=metrics[f"R_{k_val}_unseen"],

                f1=metrics[f"F1_{k_val}_unseen"],

            )

        )

        glob_line.append(

            "P_{k}_glob {p:.4f} | R_{k}_glob {r:.4f} | F1_{k}_glob {f1:.4f}".format(

                k=k_val,

                p=metrics[f"P_{k_val}_glob"],

                r=metrics[f"R_{k_val}_glob"],

                f1=metrics[f"F1_{k_val}_glob"],

            )

        )

    logger.info(" | ".join(unseen_line))

    logger.info(" | ".join(glob_line))

    return metrics





def evaluate_mssa(                             

    data_loader,

    model,

    cfg,

    logger,

    num_seen_classes=None,

    report_dir=None,

    unseen_class_names=None,

    all_class_names=None,

):

    model.eval()

    preds_unseen = []

    preds_glob = []

    labels_unseen = []

    labels_glob = []

    model_module = get_model_module(model)

    tta_info = get_eval_tta_info(model_module, cfg)                                  

    log_eval_tta_info(logger, tta_info)                            

    rerank_info = get_eval_rerank_info(model_module, cfg)                   

    log_eval_rerank_info(logger, rerank_info)                               

    class_rerank_info = get_eval_class_rerank_info(cfg)                         

    log_eval_class_rerank_info(logger, class_rerank_info)                       

    chunk_info = get_eval_chunk_info(model_module, cfg)                       

    log_eval_chunk_info(logger, chunk_info)                          

    text_feats = None                                                                   

    effective_num_seen_classes = num_seen_classes                    

    relative_knn_confuser_idx = None                                                               

    patch_features = []                                                                                 

    visual_feats = []                                                                                                      



    with torch.no_grad():

        if not (tta_info["enabled"] and tta_info["scope"] == "unseen_only" and effective_num_seen_classes is None):     

            with autocast():

                text_feats = model_module.encode_text_features(                                                 

                    use_tta=tta_info["enabled"],

                    num_seen_classes=effective_num_seen_classes,

                )

            if class_rerank_info["enabled"] and effective_num_seen_classes is not None:     

                relative_knn_confuser_idx = _build_relative_knn_confuser_index(                                      

                    text_feats,

                    class_rerank_info,

                    effective_num_seen_classes,

                )



    for images, seen_labels, unseen_labels, img_list in tqdm(data_loader):

        device = get_device()

        images = images.to(device)               

        num_seen_cls = seen_labels.size(1)                     

        if effective_num_seen_classes is None:                                                          

            effective_num_seen_classes = int(num_seen_cls)

        if text_feats is None:                        

            with torch.no_grad():

                with autocast():

                    text_feats = model_module.encode_text_features(

                        use_tta=tta_info["enabled"],

                        num_seen_classes=num_seen_cls,

                    )

        elif effective_num_seen_classes is not None and num_seen_cls != effective_num_seen_classes:              

            raise ValueError(

                f"Inconsistent num_seen_classes between dataset ({effective_num_seen_classes}) and batch ({num_seen_cls})"

            )

        if class_rerank_info["enabled"] and relative_knn_confuser_idx is None:                                                    

            relative_knn_confuser_idx = _build_relative_knn_confuser_index( 

                text_feats,

                class_rerank_info,

                effective_num_seen_classes,

            )



        output_list = []                          

        with torch.no_grad():

            with autocast():

                output = model_module.forward_for_test(                                                                            

                    images,

                    text_feats,

                    num_seen_classes=num_seen_cls,

                    return_rerank_bundle=class_rerank_info["enabled"],

                )

                for img in img_list:

                    output_ = model_module.forward_for_test(                                       

                        img.to(device),

                        text_feats,

                        num_seen_classes=num_seen_cls,

                    )

                    output_list.append(output_)

                if class_rerank_info["enabled"]:

                    output_bundle = model_module.aggregatorplus(                                   

                        output,

                        output_list,

                        count_list=cfg.count_list,

                        return_rerank_bundle=True,

                    )

                    output_final = output_bundle["logit_final"]                               

                else:

                    output_final = model_module.aggregatorplus(output, output_list, count_list=cfg.count_list)

                output_final = model_module.apply_eval_proto_rerank(                                                                        

                    output_final, text_feats, num_seen_classes=num_seen_cls

                )

                output_final = model_module.apply_eval_label_chunking(                  

                    output_final, num_seen_classes=num_seen_cls

                )



        logits_unseen = output_final[:, num_seen_cls:]                        

        logits_glob = output_final                         



        preds_unseen.append(logits_unseen.float().cpu())                               

        preds_glob.append(logits_glob.float().cpu())                               

        labels_unseen.append(unseen_labels)                                        

        labels_glob.append(torch.cat([seen_labels, unseen_labels], dim=1))                              

        if class_rerank_info["enabled"]:                                       

            patch_topk = max(1, min(int(class_rerank_info["patch_topk"]), output_bundle["logit_local"].shape[2]))                                              

            local_topk = torch.topk(output_bundle["logit_local"], k=patch_topk, dim=2).values.mean(dim=2)                           

            local_mean = output_bundle["logit_local"].mean(dim=2)            

            local_gap = local_mean - output_bundle["logit_glob"]                                                          

            patch_feature = (                                     

                float(class_rerank_info["patch_local_blend"]) * (local_topk - local_mean)                    

                + (1.0 - float(class_rerank_info["patch_local_blend"])) * local_gap                         

            )

            patch_features.append(patch_feature.float().cpu())                                                                

            visual_feats.append(output_bundle["visual_feature"].float().cpu())  



    preds_unseen = torch.cat(preds_unseen, dim=0)                                                                    

    preds_glob = torch.cat(preds_glob, dim=0)                                                                                        

    labels_unseen = torch.cat(labels_unseen, dim=0)

    labels_glob = torch.cat(labels_glob, dim=0)

    if class_rerank_info["enabled"]:

        if torch.cuda.is_available():      

            model.to(torch.device("cpu"))

            torch.cuda.empty_cache()

        preds_unseen, preds_glob = apply_eval_classwise_rerank(                                            

            preds_glob,

            torch.cat(patch_features, dim=0),

            torch.cat(visual_feats, dim=0),

            class_rerank_info,

            num_seen_classes=effective_num_seen_classes,

            relative_knn_confuser_idx=relative_knn_confuser_idx,

        )



    unseen_mask = (labels_unseen > 0).sum(1) > 0

    unseen_metrics = compute_metric_block(

        preds_unseen[unseen_mask],

        labels_unseen[unseen_mask],

        "unseen",

        class_names=unseen_class_names,

        report_dir=report_dir,

        dataset_name=cfg.DATASET.NAME,

    )



    glob_mask = (labels_glob > 0).sum(1) > 0

    glob_metrics = compute_metric_block(

        preds_glob[glob_mask],

        labels_glob[glob_mask],

        "glob",

        class_names=all_class_names,

        report_dir=report_dir,

        dataset_name=cfg.DATASET.NAME,

    )



    metrics = {**unseen_metrics, **glob_metrics}

    topk_list = get_eval_topk_list(cfg.DATASET.NAME)

    logger.info("Evaluating predictions over all images")

    unseen_line = [f"mAP_unseen {metrics['mAP_unseen']:.2f}"]

    glob_line = [f"mAP_glob {metrics['mAP_glob']:.2f}"]

    for k_val in topk_list:

        unseen_line.append(

            "P_{k}_unseen {p:.4f} | R_{k}_unseen {r:.4f} | F1_{k}_unseen {f1:.4f}".format(

                k=k_val,

                p=metrics[f"P_{k_val}_unseen"],

                r=metrics[f"R_{k_val}_unseen"],

                f1=metrics[f"F1_{k_val}_unseen"],

            )

        )

        glob_line.append(

            "P_{k}_glob {p:.4f} | R_{k}_glob {r:.4f} | F1_{k}_glob {f1:.4f}".format(

                k=k_val,

                p=metrics[f"P_{k_val}_glob"],

                r=metrics[f"R_{k_val}_glob"],

                f1=metrics[f"F1_{k_val}_glob"],

            )

        )

    logger.info(" | ".join(unseen_line))

    logger.info(" | ".join(glob_line))

    return metrics





def main():                              

    global args            

    args = parser.parse_args()          

    args.config_file = resolve_config_file(args)                                 

    cfg = setup_cfg(args)                                                      

    setup_seed(cfg.SEED)         

    cudnn.benchmark = True                                         



    checkpoint_path = Path(args.checkpoint)                                      

    if not checkpoint_path.is_file():

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")



    if args.output_dir is not None:

        output_root = Path(args.output_dir)

    else:

        output_root = checkpoint_path.parent



    record_name = datetime.datetime.now().strftime("%m-%d-%H-%M-%S") + "_EVAL"                 

    record_path = output_root / record_name

    record_path.mkdir(parents=True, exist_ok=True)



    cfg.defrost()                           

    cfg.RECORD_PATH = str(record_path)

    cfg.freeze()



    logger = init_log(str(record_path))                                                       

    write_description_to_folder(record_path / "configs.txt", cfg)                                                                                   

    environment_info = save_reproducibility_metadata(

        record_path,

        checkpoint_path=checkpoint_path,

        config_path=args.config_file,

    )

    logger.info("Config path: %s", args.config_file)               

    logger.info("Checkpoint path: %s", checkpoint_path)                



    test_dataset = build_dataset(cfg, cfg.DATASET.TEST_SPLIT)                                                       

    seen_cls_names = test_dataset.seen_names                                                    

    unseen_cls_names = test_dataset.unseen_names        

    if isinstance(seen_cls_names, np.ndarray):

        seen_cls_names = seen_cls_names.tolist()

        unseen_cls_names = unseen_cls_names.tolist()

    all_cls_names = seen_cls_names + unseen_cls_names



    test_loader = DataLoader(                                   

        test_dataset,

        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,

        shuffle=cfg.DATALOADER.TEST.SHUFFLE,

        num_workers=cfg.DATALOADER.NUM_WORKERS,

        pin_memory=True,

        drop_last=False,

    )



    model = build_text_model(cfg, all_cls_names)                                                                                 

    checkpoint_obj = safe_torch_load(str(checkpoint_path))                                               

    prompt_state, checkpoint_meta = extract_prompt_state(checkpoint_obj)                                                                                                         



    prompt_learner = get_prompt_learner(model)                                                                                       

    load_result = prompt_learner.load_state_dict(prompt_state, strict=False)

    logger.info("Checkpoint type: %s", checkpoint_meta.get("checkpoint_type", "unknown"))           

    if checkpoint_meta.get("checkpoint_epoch") is not None:

        logger.info("Checkpoint epoch: %s", checkpoint_meta["checkpoint_epoch"])

    if checkpoint_meta.get("best_unseen_F1") is not None:

        logger.info("Checkpoint best_F1_3_unseen: %.4f", checkpoint_meta["best_unseen_F1"])

    if checkpoint_meta.get("best_unseen_mAP") is not None:

        logger.info("Checkpoint best_mAP_unseen: %.4f", checkpoint_meta["best_unseen_mAP"])

    if checkpoint_meta.get("best_gzsl_F1") is not None:

        logger.info("Checkpoint best_F1_3_global: %.4f", checkpoint_meta["best_gzsl_F1"])

    if checkpoint_meta.get("best_unseen_epoch") is not None:

        logger.info("Checkpoint best_F1_3_unseen epoch: %s", checkpoint_meta["best_unseen_epoch"])

    if checkpoint_meta.get("best_unseen_mAP_epoch") is not None:

        logger.info("Checkpoint best_mAP_unseen epoch: %s", checkpoint_meta["best_unseen_mAP_epoch"])

    if checkpoint_meta.get("best_gzsl_epoch") is not None:

        logger.info("Checkpoint best_F1_3_global epoch: %s", checkpoint_meta["best_gzsl_epoch"])

    if checkpoint_meta.get("best_monitor_name") is not None and checkpoint_meta.get("best_monitor_value") is not None:

        logger.info(

            "Checkpoint best_%s: %.4f (epoch=%s)",

            checkpoint_meta["best_monitor_name"],

            checkpoint_meta["best_monitor_value"],

            checkpoint_meta.get("best_monitor_epoch"),

        )

    logger.info("Prompt learner load result: missing=%s unexpected=%s", load_result.missing_keys, load_result.unexpected_keys)



    if "clsn" in load_result.missing_keys:

        raise RuntimeError("Checkpoint does not contain the essential prompt learner parameter 'clsn'")



    if args.use_dataparallel and torch.cuda.device_count() > 1:

        logger.info("Wrapping model with DataParallel across %d GPUs", torch.cuda.device_count())

        model = nn.DataParallel(model)



    if cfg.cutimage:                     

        metrics = evaluate_mssa(                   

            test_loader,

            model,

            cfg,

            logger,

            num_seen_classes=len(seen_cls_names),

            report_dir=record_path,

            unseen_class_names=unseen_cls_names,

            all_class_names=all_cls_names,

        )

    else:                 

        metrics = evaluate_standard(

            test_loader,

            model,

            logger,

            num_seen_classes=len(seen_cls_names),

            report_dir=record_path,

            unseen_class_names=unseen_cls_names,

            all_class_names=all_cls_names,

        )



    result_payload = {

        "checkpoint_path": str(checkpoint_path),

        "checkpoint_sha256": environment_info["checkpoint_sha256"],

        "config_path": str(Path(args.config_file).resolve()),

        "command_path": str(record_path / "command.txt"),

        "environment_path": str(record_path / "environment.json"),

        "checkpoint_type": checkpoint_meta.get("checkpoint_type"),

        "checkpoint_epoch": to_serializable(checkpoint_meta.get("checkpoint_epoch")),

        "best_unseen_F1": to_serializable(checkpoint_meta.get("best_unseen_F1")),

        "best_unseen_mAP": to_serializable(checkpoint_meta.get("best_unseen_mAP")),

        "best_gzsl_F1": to_serializable(checkpoint_meta.get("best_gzsl_F1")),

        "best_unseen_epoch": to_serializable(checkpoint_meta.get("best_unseen_epoch")),

        "best_unseen_mAP_epoch": to_serializable(checkpoint_meta.get("best_unseen_mAP_epoch")),

        "best_gzsl_epoch": to_serializable(checkpoint_meta.get("best_gzsl_epoch")),

        "best_monitor_name": to_serializable(checkpoint_meta.get("best_monitor_name")),

        "best_monitor_value": to_serializable(checkpoint_meta.get("best_monitor_value")),

        "best_monitor_epoch": to_serializable(checkpoint_meta.get("best_monitor_epoch")),

        "cutimage": bool(cfg.cutimage),

        "count_list": list(cfg.count_list),

        "tta_enabled": bool(cfg.TTA.ENABLED),

        "tta_mode": str(cfg.TTA.MODE),

        "tta_scope": str(getattr(cfg.TTA, "SCOPE", "all")),

        "tta_include_base_prompt": bool(getattr(cfg.TTA, "INCLUDE_BASE_PROMPT", False)),

        "tta_aggregation": str(cfg.TTA.AGGREGATION),

        "tta_base_weight": float(getattr(cfg.TTA, "BASE_WEIGHT", 1.0)),

        "tta_template_weights": list(getattr(cfg.TTA, "TEMPLATE_WEIGHTS", [])),

        "tta_num_templates": int(len(cfg.TTA.PROMPT_TEMPLATES)),

        "rerank_enabled": bool(getattr(cfg.RERANK, "ENABLED", False)),

        "rerank_mode": str(getattr(cfg.RERANK, "MODE", "disabled")),

        "rerank_scope": str(getattr(cfg.RERANK, "SCOPE", "all")),

        "rerank_topk": int(getattr(cfg.RERANK, "TOPK", 0)),

        "rerank_sim_temperature": float(getattr(cfg.RERANK, "SIM_TEMPERATURE", 0.2)),

        "rerank_logit_temperature": float(getattr(cfg.RERANK, "LOGIT_TEMPERATURE", 1.0)),

        "rerank_gate_mode": str(getattr(cfg.RERANK, "GATE_MODE", "none")),

        "rerank_gate_topk": int(getattr(cfg.RERANK, "GATE_TOPK", 0)),

        "rerank_gate_threshold": float(getattr(cfg.RERANK, "GATE_THRESHOLD", 0.0)),

        "class_rerank_enabled": bool(getattr(cfg.CLASS_RERANK, "ENABLED", False)),

        "class_rerank_scope": str(getattr(cfg.CLASS_RERANK, "SCOPE", "unseen_only")),

        "class_rerank_visual_topk": int(getattr(cfg.CLASS_RERANK, "VISUAL_TOPK", 20)),

        "class_rerank_visual_chunk_size": int(getattr(cfg.CLASS_RERANK, "VISUAL_CHUNK_SIZE", 256)),

        "class_rerank_visual_temperature": float(getattr(cfg.CLASS_RERANK, "VISUAL_TEMPERATURE", 0.10)),

        "class_rerank_patch_topk": int(getattr(cfg.CLASS_RERANK, "PATCH_TOPK", 4)),

        "class_rerank_patch_weight": float(getattr(cfg.CLASS_RERANK, "PATCH_WEIGHT", 0.10)),

        "class_rerank_margin_weight": float(getattr(cfg.CLASS_RERANK, "MARGIN_WEIGHT", 0.08)),

        "class_rerank_knn_weight": float(getattr(cfg.CLASS_RERANK, "KNN_WEIGHT", 0.18)),

        "class_rerank_relative_knn_enabled": bool(getattr(cfg.CLASS_RERANK, "RELATIVE_KNN_ENABLED", False)),

        "class_rerank_relative_knn_topk": int(getattr(cfg.CLASS_RERANK, "RELATIVE_KNN_TOPK", 3)),

        "class_rerank_relative_knn_weight": float(getattr(cfg.CLASS_RERANK, "RELATIVE_KNN_WEIGHT", 0.05)),

        "class_rerank_patch_local_blend": float(getattr(cfg.CLASS_RERANK, "PATCH_LOCAL_BLEND", 0.5)),

        "class_rerank_zscore_clamp": float(getattr(cfg.CLASS_RERANK, "ZSCORE_CLAMP", 5.0)),

        "class_rerank_knn_score_chunk_size": int(getattr(cfg.CLASS_RERANK, "KNN_SCORE_CHUNK_SIZE", 2048)),

        "chunk_enabled": bool(getattr(cfg.CHUNK, "ENABLED", False)),

        "chunk_mode": str(getattr(cfg.CHUNK, "MODE", "disabled")),

        "chunk_scope": str(getattr(cfg.CHUNK, "SCOPE", "all")),

        "chunk_size": int(getattr(cfg.CHUNK, "SIZE", 0)),

        "chunk_topk_per_chunk": int(getattr(cfg.CHUNK, "TOPK_PER_CHUNK", 0)),

        "chunk_filter_value": float(getattr(cfg.CHUNK, "FILTER_VALUE", -1e4)),

        "chunk_balanced": bool(getattr(cfg.CHUNK, "BALANCED", True)),

        **metrics,

    }



    with open(record_path / "metrics.json", "w", encoding="utf-8") as fp:

        json.dump(result_payload, fp, indent=2)



    logger.info("Full evaluation summary\n%s", format_metrics(result_payload, cfg.DATASET.NAME))

    logger.info("Saved metrics to %s", record_path / "metrics.json")





if __name__ == "__main__":  

    main()

