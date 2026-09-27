import os

import json

import pickle

import argparse

import datetime

import subprocess

import time

import torch

import numpy as np

import torch.nn as nn

import torch.backends.cudnn as cudnn



from torch.utils.data import DataLoader

from torch.optim import lr_scheduler

from utils.build_cfg import setup_cfg

from utils.helper import *

from utils.lr_scheduler import build_lr_scheduler

from datasets import Caption, build_dataset

from model import build_text_model

from utils.trainer_text import train, validate, validate_mssa



                                                 

      

parser = argparse.ArgumentParser(description='PyTorch-Caption_Training')

parser.add_argument('--config_file', dest='config_file', type=str, help='model config file path')

parser.add_argument('--output_dir', type=str, default=None)

parser.add_argument(

    "opts",

    default=None,

    nargs=argparse.REMAINDER,

    help="Modify config options using the command line",

)



                                                                      

def get_prompt_learner(model):

    if isinstance(model, nn.DataParallel):

        return model.module.prompt_learner

    return model.prompt_learner



                                                           

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

    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):

        return normalize_prompt_state_dict(checkpoint_obj["state_dict"])

    if isinstance(checkpoint_obj, dict):

        return normalize_prompt_state_dict(checkpoint_obj)

    raise ValueError("Unsupported init checkpoint format")



                                                            

def load_init_weights(model, init_path, logger):

    if not init_path:

        return

    if not os.path.exists(init_path):

        raise FileNotFoundError(f"Init weights not found: {init_path}")



    logger.info("... loading init prompt weights from %s", init_path)

    try:

        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)

    except TypeError:

        checkpoint = torch.load(init_path, map_location="cpu")

    prompt_state = extract_prompt_state(checkpoint)

    load_result = get_prompt_learner(model).load_state_dict(prompt_state, strict=False)

    logger.info(

        "Init prompt load result: missing=%s unexpected=%s",

        load_result.missing_keys,

        load_result.unexpected_keys,

    )

    if "clsn" in load_result.missing_keys:

        raise RuntimeError("Init checkpoint does not contain the essential prompt learner parameter 'clsn'")



                                                                 

def resolve_monitor_value(cfg, metrics):

    primary_topk = get_primary_eval_topk(cfg)

    primary_unseen_key, primary_glob_key = get_primary_f1_metric_names(cfg)

    monitor_name = cfg.TRAIN.EARLY_STOP.MONITOR.lower()

    if monitor_name in {

        "f1_3_unseen",

        "f1_unseen",

        "best_f1_3_unseen",

        f"f1_{primary_topk}_unseen",

        f"best_f1_{primary_topk}_unseen",

    }:

        return float(metrics[primary_unseen_key]), primary_unseen_key

    if monitor_name in {

        "f1_3_global",

        "f1_global",

        "f1_gzsl",

        "best_f1_3_global",

        f"f1_{primary_topk}_glob",

        f"best_f1_{primary_topk}_glob",

        f"f1_{primary_topk}_global",

        f"best_f1_{primary_topk}_global",

    }:

        return float(metrics[primary_glob_key]), primary_glob_key

    if monitor_name in {"map_unseen", "best_map_unseen"}:

        return float(metrics["mAP_unseen"]), "mAP_unseen"

    if monitor_name in {"map_global", "map_glob", "map_gzsl", "best_map_global"}:

        return float(metrics["mAP_glob"]), "mAP_glob"

    raise ValueError(f"Unsupported early-stop monitor: {cfg.TRAIN.EARLY_STOP.MONITOR}")



                                                

def as_scalar(value, default=0.0):

    if value is None:

        return default

    if torch.is_tensor(value):

        return float(value.item())

    return float(value)





def get_primary_eval_topk(cfg):

    dataset_name = str(getattr(cfg.DATASET, "NAME", "")).lower()

    if dataset_name == "openimagesv4":

        return 10

    return 3





def get_primary_f1_metric_names(cfg):

    topk = get_primary_eval_topk(cfg)

    return f"F1_{topk}_unseen", f"F1_{topk}_glob"





def to_serializable(value):

    if torch.is_tensor(value):

        if value.numel() == 1:

            return value.item()

        return value.tolist()

    if isinstance(value, np.ndarray):

        return value.tolist()

    return value





def build_eval_payload(cfg, metrics, save_dict, checkpoint_name):

    payload = {

        "checkpoint_path": os.path.join(cfg.RECORD_PATH, checkpoint_name),

        "checkpoint_type": "full_checkpoint",

        "checkpoint_epoch": save_dict.get("epoch"),

        "best_unseen_F1": to_serializable(save_dict.get("best_unseen_F1")),

        "best_unseen_mAP": to_serializable(save_dict.get("best_unseen_mAP")),

        "best_gzsl_F1": to_serializable(save_dict.get("best_gzsl_F1")),

        "best_unseen_epoch": save_dict.get("best_unseen_epoch"),

        "best_unseen_mAP_epoch": save_dict.get("best_unseen_mAP_epoch"),

        "best_gzsl_epoch": save_dict.get("best_gzsl_epoch"),

        "best_monitor_name": save_dict.get("best_monitor_name"),

        "best_monitor_value": to_serializable(save_dict.get("best_monitor_value")),

        "best_monitor_epoch": save_dict.get("best_monitor_epoch"),

        "cutimage": bool(cfg.cutimage),

        "count_list": list(cfg.count_list),

        "tta_enabled": bool(getattr(cfg.TTA, "ENABLED", False)),

        "tta_mode": str(getattr(cfg.TTA, "MODE", "disabled")),

        "tta_scope": str(getattr(cfg.TTA, "SCOPE", "all")),

        "tta_include_base_prompt": bool(getattr(cfg.TTA, "INCLUDE_BASE_PROMPT", False)),

        "tta_aggregation": str(getattr(cfg.TTA, "AGGREGATION", "none")),

        "tta_base_weight": float(getattr(cfg.TTA, "BASE_WEIGHT", 1.0)),

        "tta_template_weights": [float(x) for x in getattr(cfg.TTA, "TEMPLATE_WEIGHTS", [])],

        "tta_num_templates": len(getattr(cfg.TTA, "PROMPT_TEMPLATES", [])),

        "rerank_enabled": bool(getattr(cfg.RERANK, "ENABLED", False)),

        "rerank_mode": str(getattr(cfg.RERANK, "MODE", "disabled")),

        "rerank_scope": str(getattr(cfg.RERANK, "SCOPE", "all")),

        "rerank_topk": int(getattr(cfg.RERANK, "TOPK", 0)),

        "rerank_sim_temperature": float(getattr(cfg.RERANK, "SIM_TEMPERATURE", 0.2)),

        "rerank_logit_temperature": float(getattr(cfg.RERANK, "LOGIT_TEMPERATURE", 1.0)),

        "chunk_enabled": bool(getattr(cfg.CHUNK, "ENABLED", False)),

        "chunk_mode": str(getattr(cfg.CHUNK, "MODE", "disabled")),

        "chunk_scope": str(getattr(cfg.CHUNK, "SCOPE", "all")),

        "chunk_size": int(getattr(cfg.CHUNK, "SIZE", 0)),

        "chunk_topk_per_chunk": int(getattr(cfg.CHUNK, "TOPK_PER_CHUNK", 0)),

        "chunk_filter_value": float(getattr(cfg.CHUNK, "FILTER_VALUE", -1e4)),

        "chunk_balanced": bool(getattr(cfg.CHUNK, "BALANCED", True)),

        "oracle_mode": bool(getattr(cfg.EXP, "ORACLE_MODE", False)),

        "oracle_text_mode": str(getattr(cfg.EXP, "ORACLE_TEXT_MODE", "")),

        "oracle_text_template": str(getattr(cfg.EXP, "ORACLE_TEXT_TEMPLATE", "")),

        "oracle_max_labels_per_sample": int(getattr(cfg.EXP, "ORACLE_MAX_LABELS_PER_SAMPLE", 0)),

        "oracle_include_base_captions": bool(getattr(cfg.EXP, "ORACLE_INCLUDE_BASE_CAPTIONS", False)),

        "oracle_sample_limit": int(getattr(cfg.EXP, "ORACLE_SAMPLE_LIMIT", 0)),

        "oracle_proto_weight": float(getattr(cfg.EXP, "ORACLE_PROTO_WEIGHT", 0.0)),

        "oracle_bce_weight": float(getattr(cfg.EXP, "ORACLE_BCE_WEIGHT", 0.0)),

        "oracle_primary_mse_weight": float(getattr(cfg.EXP, "ORACLE_PRIMARY_MSE_WEIGHT", 0.0)),

        "extra_caption_json": str(getattr(cfg.EXP, "EXTRA_CAPTION_JSON", "")),

        "train_class_filter_json": str(getattr(cfg.EXP, "TRAIN_CLASS_FILTER_JSON", "")),

        "ot_enabled": bool(getattr(cfg.OT, "ENABLED", False)),

        "ot_mode": str(getattr(cfg.OT, "MODE", "disabled")),

        "hard_neg_enabled": bool(getattr(cfg.TRAIN.HARD_NEG, "ENABLED", False)),

        "train_batch_size": int(getattr(cfg.DATALOADER.TRAIN_X, "BATCH_SIZE", 0)),

        "train_accum_steps": int(getattr(cfg.TRAIN, "ACCUM_STEPS", 1)),

        "hard_neg_weight": float(getattr(cfg.TRAIN.HARD_NEG, "WEIGHT", 0.0)),

        "hard_neg_margin": float(getattr(cfg.TRAIN.HARD_NEG, "MARGIN", 0.0)),

        "hard_neg_topk": int(getattr(cfg.TRAIN.HARD_NEG, "TOPK", 0)),

        "hard_neg_start_epoch": int(getattr(cfg.TRAIN.HARD_NEG, "START_EPOCH", 0)),

        "hard_neg_pool": str(getattr(cfg.TRAIN.HARD_NEG, "POOL", "disabled")),

        "hard_neg_neighbor_topk": int(getattr(cfg.TRAIN.HARD_NEG, "NEIGHBOR_TOPK", 0)),

    }

    payload.update({key: to_serializable(value) for key, value in metrics.items()})

    return payload





def save_eval_payload(path, payload):

    with open(path, "w", encoding="utf-8") as fp:

        json.dump(payload, fp, indent=2, ensure_ascii=False)





def query_gpu_free_mib():

    try:

        output = subprocess.check_output(

            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],

            text=True,

        ).strip().splitlines()

    except Exception:

        return None

    if not output:

        return None

    try:

        return int(output[0].strip())

    except ValueError:

        return None





def wait_for_eval_window(cfg, logger):

    min_free = int(getattr(cfg.TRAIN, "EVAL_WAIT_FREE_MIB", 0) or 0)

    if min_free <= 0 or not torch.cuda.is_available():

        return



    retry_sec = int(getattr(cfg.TRAIN, "EVAL_WAIT_RETRY_SEC", 60) or 60)

    while True:

        free_mib = query_gpu_free_mib()

        if free_mib is not None and free_mib >= min_free:

            logger.info(

                "Eval GPU window ready: free memory %d MiB >= %d MiB.",

                free_mib,

                min_free,

            )

            return

        logger.info(

            "Eval waiting for GPU window: free memory %s MiB < %d MiB, sleeping %ds.",

            free_mib if free_mib is not None else "unknown",

            min_free,

            retry_sec,

        )

        time.sleep(retry_sec)



                                               

def main():

    global args                    

    args = parser.parse_args()                  

    cfg = setup_cfg(args)           

    setup_seed(cfg.SEED)            

    cudnn.benchmark = True        



             

                                                

    if args.output_dir is not None:

        record_path = cfg.OUTPUT_DIR

    else:

        record_name = datetime.datetime.now().strftime('%m-%d-%H-%M-%S') + "_" + "RENAME"

        record_path = os.path.join(cfg.OUTPUT_DIR, record_name)

    cfg.defrost()

    cfg.RECORD_PATH = record_path

    cfg.freeze()

    os.makedirs(record_path, exist_ok=True)

    logger = init_log(record_path)

    write_description_to_folder(os.path.join(record_path, "configs.txt"), cfg)



                                                     

    test_dataset = build_dataset(cfg, cfg.DATASET.TEST_SPLIT)

    seen_cls_names = test_dataset.seen_names

    unseen_cls_names = test_dataset.unseen_names

    

                                                          

    if isinstance(seen_cls_names, np.ndarray):

        seen_cls_names = seen_cls_names.tolist()

        unseen_cls_names = unseen_cls_names.tolist()

    all_cls_names = seen_cls_names + unseen_cls_names                      



    train_dataset = Caption(

        text_path=cfg.DATASET.TEXT_PATH,

        dataset=cfg.DATASET.NAME,

        extra_template=bool(getattr(cfg.TRAINER.TEXT, "EXTRA_TEMPLATE", True)),

        cfg=cfg,

        classnames=all_cls_names,

    )

    

    train_loader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,

                              shuffle=cfg.DATALOADER.TRAIN_X.SHUFFLE, num_workers=cfg.DATALOADER.NUM_WORKERS, 

                              pin_memory=True, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,

                             shuffle=cfg.DATALOADER.TEST.SHUFFLE, num_workers=cfg.DATALOADER.NUM_WORKERS,

                             pin_memory=True, drop_last=False)

    

                                                    

    model = build_text_model(cfg, all_cls_names)                                     

    if cfg.MODEL.INIT_WEIGHTS:                     

        load_init_weights(model, cfg.MODEL.INIT_WEIGHTS, logger)

    device_count = torch.cuda.device_count()

    if device_count > 1:                             

        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")

        model = nn.DataParallel(model)

    

                                                     

                                       

    try:                        

        param_group = []

        for name, param in model.named_parameters():

            if param.requires_grad:

                logger.info(name)

                param_group.append(param)

    except:                        

        param_group = model.module.parameters()

    

             

                                  

    optim = torch.optim.SGD(param_group, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM, weight_decay=cfg.OPTIM.WEIGHT_DECAY)

    sched = build_lr_scheduler(optim, cfg.OPTIM)            



                                                       

    best_unseen_F1 = 0

    best_unseen_mAP = 0

    best_gzsl_F1 = 0

    best_unseen_epoch = 0

    best_unseen_mAP_epoch = 0

    best_gzsl_epoch = 0

    best_monitor_value = float("-inf")

    best_monitor_epoch = 0

    epochs_since_improvement = 0

    start_epoch = 0

    if cfg.RESUME is not None:                                  

        if os.path.exists(cfg.RESUME):

            logger.info('... loading pretrained weights from %s' % cfg.RESUME)

            try:

                checkpoint = torch.load(cfg.RESUME, map_location='cpu', weights_only=False)

            except TypeError:

                checkpoint = torch.load(cfg.RESUME, map_location='cpu')

            start_epoch = checkpoint['epoch']

            best_unseen_F1 = as_scalar(checkpoint.get('best_unseen_F1', 0))

            best_unseen_mAP = as_scalar(checkpoint.get('best_unseen_mAP', 0))

            best_gzsl_F1 = as_scalar(checkpoint.get('best_gzsl_F1', 0))

            best_unseen_epoch = int(checkpoint.get('best_unseen_epoch', 0) or 0)

            best_unseen_mAP_epoch = int(checkpoint.get('best_unseen_mAP_epoch', 0) or 0)

            best_gzsl_epoch = int(checkpoint.get('best_gzsl_epoch', 0) or 0)

            best_monitor_value = as_scalar(checkpoint.get('best_monitor_value', best_unseen_F1), best_unseen_F1)

            best_monitor_epoch = int(checkpoint.get('best_monitor_epoch', best_unseen_epoch) or 0)

            epochs_since_improvement = int(checkpoint.get('epochs_since_improvement', 0) or 0)

            get_prompt_learner(model).load_state_dict(checkpoint['state_dict'])

            optim.load_state_dict(checkpoint['optimizer'])

            sched.load_state_dict(checkpoint['scheduler'])

    

                                                     

    for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

                                                        

        batch_time, losses = train(train_loader, model, optim, sched, cfg, logger, epoch)

        logger.info('Train: [{0}/{1}]\t'

                    'Time {batch_time.avg:.3f}\t'

                    'Loss {losses.avg:.2f}\t'.format(

                    epoch+1, cfg.OPTIM.MAX_EPOCH, batch_time=batch_time, losses=losses))

              

        if (epoch+1) % cfg.TRAIN.EVAL_PERIOD == 0 or epoch == cfg.OPTIM.MAX_EPOCH - 1:

            wait_for_eval_window(cfg, logger)

                  

            if cfg.cutimage is False:   

                metrics = validate(test_loader, model, cfg, logger, epoch)

            else:       

                metrics = validate_mssa(test_loader, model, cfg, logger, epoch)

            

                    

            primary_unseen_metric, primary_glob_metric = get_primary_f1_metric_names(cfg)

            f1_unseen = metrics[primary_unseen_metric]

            f1_gzsl = metrics[primary_glob_metric]

            map_unseen = metrics["mAP_unseen"]

            

                    

            is_unseen_best = f1_unseen > best_unseen_F1

            if is_unseen_best:

                best_unseen_F1 = f1_unseen

                best_unseen_epoch = epoch + 1



            is_unseen_map_best = map_unseen > best_unseen_mAP

            if is_unseen_map_best:

                best_unseen_mAP = map_unseen

                best_unseen_mAP_epoch = epoch + 1



            is_gzsl_best = f1_gzsl > best_gzsl_F1

            if is_gzsl_best:

                best_gzsl_F1 = f1_gzsl

                best_gzsl_epoch = epoch + 1

            

                    

            monitor_value, monitor_name = resolve_monitor_value(cfg, metrics)

            min_delta = float(cfg.TRAIN.EARLY_STOP.MIN_DELTA)

            is_monitor_best = monitor_value > (best_monitor_value + min_delta)

            if is_monitor_best:

                best_monitor_value = monitor_value

                best_monitor_epoch = epoch + 1

                epochs_since_improvement = 0

            else:

                epochs_since_improvement += 1



                                                                     

            save_dict = {'epoch': epoch + 1,

                         'state_dict': get_prompt_learner(model).state_dict(),

                         'best_unseen_F1': best_unseen_F1,

                         'best_unseen_mAP': best_unseen_mAP,

                         'best_gzsl_F1': best_gzsl_F1,

                         'best_unseen_epoch': best_unseen_epoch,

                         'best_unseen_mAP_epoch': best_unseen_mAP_epoch,

                         'best_gzsl_epoch': best_gzsl_epoch,

                         'best_monitor_name': monitor_name,

                         'best_monitor_value': best_monitor_value,

                         'best_monitor_epoch': best_monitor_epoch,

                         'epochs_since_improvement': epochs_since_improvement,

                         'eval_metrics': metrics,

                         'optimizer': optim.state_dict(),

                         'scheduler': sched.state_dict()

                         }

            

                                                        

            save_checkpoint(save_dict, is_monitor_best, record_path)

            save_checkpoint(save_dict, is_unseen_best, record_path, prefix='unseen')

            save_checkpoint(save_dict, is_unseen_map_best, record_path, prefix='map_unseen')

            save_checkpoint(save_dict, is_gzsl_best, record_path, prefix='gzsl')

            latest_payload = build_eval_payload(cfg, metrics, save_dict, "checkpoint.pth.tar")

            save_eval_payload(os.path.join(record_path, "latest_metrics.json"), latest_payload)

            if is_monitor_best:

                best_payload = build_eval_payload(cfg, metrics, save_dict, "model_best.pth.tar")

                save_eval_payload(os.path.join(record_path, "best_metrics.json"), best_payload)

            logger.info(      

                ' * best_{metric1}={best1:.4f} (epoch={epoch1})\t'

                'best_mAP_unseen={best2:.4f} (epoch={epoch2})\t'

                'best_{metric3}={best3:.4f} (epoch={epoch3})\t'

                'best_{monitor}={best4:.4f} (epoch={epoch4})\t'

                'epochs_since_improvement={bad_epochs}'.format(

                    metric1=primary_unseen_metric,

                    best1=best_unseen_F1,

                    epoch1=best_unseen_epoch,

                    best2=best_unseen_mAP,

                    epoch2=best_unseen_mAP_epoch,

                    metric3=primary_glob_metric,

                    best3=best_gzsl_F1,

                    epoch3=best_gzsl_epoch,

                    monitor=monitor_name,

                    best4=best_monitor_value,

                    epoch4=best_monitor_epoch,

                    bad_epochs=epochs_since_improvement,

                )

            )

                                                          

            if cfg.TRAIN.EARLY_STOP.ENABLED:

                patience = int(cfg.TRAIN.EARLY_STOP.PATIENCE)

                min_epochs = int(cfg.TRAIN.EARLY_STOP.MIN_EPOCHS)

                if (epoch + 1) >= min_epochs and epochs_since_improvement >= patience:

                    logger.info(

                        'Early stopping triggered at epoch %d: no %s improvement for %d eval rounds. '

                        'Best %s %.4f was reached at epoch %d.',

                        epoch + 1,

                        monitor_name,

                        epochs_since_improvement,

                        monitor_name,

                        best_monitor_value,

                        best_monitor_epoch,

                    )

                    break



       

if __name__ == '__main__':

    main()











    

