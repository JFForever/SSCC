import torch

import torch.nn.functional as F

import time

from tqdm import tqdm

from utils.helper import (

    AverageMeter,

    compute_AP,

    compute_AP_openimages_biam,

    compute_F1,

    compute_weighted_map_percent,

    filter_openimages_biam_classes,

    is_openimagesv4_dataset,

)

from utils.ot import build_similarity_source_mass, cosine_cost_matrix, masked_ot_distance

from torch.cuda.amp import autocast





def get_model_module(model):

    if isinstance(model, torch.nn.DataParallel):

        return model.module

    return model





def get_eval_tta_info(model_module, cfg):

    if hasattr(model_module, "get_eval_tta_info"):

        info = model_module.get_eval_tta_info()

    else:

        info = {

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

    return info





def get_eval_chunk_info(model_module, cfg):

    if hasattr(model_module, "get_eval_chunk_info"):

        info = model_module.get_eval_chunk_info()

    else:

        info = {

            "enabled": bool(getattr(cfg.CHUNK, "ENABLED", False)),

            "mode": "disabled",

            "scope": "all",

            "size": 0,

            "topk_per_chunk": 0,

            "filter_value": -1e4,

            "balanced": True,

        }

    return info





def get_eval_rerank_info(model_module, cfg):

    if hasattr(model_module, "get_eval_rerank_info"):

        info = model_module.get_eval_rerank_info()

    else:

        info = {

            "enabled": bool(getattr(cfg.RERANK, "ENABLED", False)),

            "mode": "disabled",

            "scope": "all",

            "topk": 0,

            "sim_temperature": 0.2,

            "logit_temperature": 1.0,

        }

    return info





def log_eval_tta_info(logger, info, stage):

    if not info["enabled"]:

        return

    logger.info(

        "%s TTA: enabled=%s mode=%s scope=%s include_base=%s aggregation=%s templates=%d",

        stage,

        info["enabled"],

        info["mode"],

        info["scope"],

        info["include_base_prompt"],

        info["aggregation"],

        info["num_templates"],

    )

    if info["aggregation"] == "weighted_feature_mean":

        logger.info(

            "%s TTA weights: base_weight=%.4f template_weights=%s",

            stage,

            float(info["base_weight"]),

            info["template_weights"],

        )





def log_eval_chunk_info(logger, info, stage):

    if not info["enabled"]:

        return

    logger.info(

        "%s Chunk: enabled=%s mode=%s scope=%s size=%d topk_per_chunk=%d balanced=%s filter_value=%.1f",

        stage,

        info["enabled"],

        info["mode"],

        info["scope"],

        int(info["size"]),

        int(info["topk_per_chunk"]),

        info["balanced"],

        float(info["filter_value"]),

    )





def log_eval_rerank_info(logger, info, stage):

    if not info["enabled"]:

        return

    logger.info(

        "%s Rerank: enabled=%s mode=%s scope=%s topk=%d sim_temp=%.3f logit_temp=%.3f",

        stage,

        info["enabled"],

        info["mode"],

        info["scope"],

        int(info["topk"]),

        float(info["sim_temperature"]),

        float(info["logit_temperature"]),

    )





def infer_num_seen_classes(data_loader):

    dataset = getattr(data_loader, "dataset", None)

    if dataset is None:

        return None



    seen_names = getattr(dataset, "seen_names", None)

    if seen_names is not None:

        return len(seen_names)



    seen_cls_idx = getattr(dataset, "seen_cls_idx", None)

    if seen_cls_idx is not None:

        return len(seen_cls_idx)



    return None





def no_similar_loss(x):

    similar = x @ x.t()

    mask = 1 - torch.eye(x.size(0))

    mask = mask.to(x.device)

    dist = (1 + similar) * mask

    return dist.sum()





def compute_hard_negative_loss(text_features, cls_embeddings, targets, cfg):

    hard_neg_cfg = cfg.TRAIN.HARD_NEG

    if not bool(getattr(hard_neg_cfg, "ENABLED", False)):

        return None



    margin = float(getattr(hard_neg_cfg, "MARGIN", 0.05))

    topk = int(getattr(hard_neg_cfg, "TOPK", 4))

    pool_mode = str(getattr(hard_neg_cfg, "POOL", "target_neighbors")).lower()

    neighbor_topk = int(getattr(hard_neg_cfg, "NEIGHBOR_TOPK", 32))

    if topk <= 0:

        return None



    logits = text_features.float() @ cls_embeddings.float().t()

    pos_scores = logits.gather(dim=1, index=targets.unsqueeze(1))



    if pool_mode == "all":

        candidate_logits = logits.clone()

        candidate_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))

    elif pool_mode == "target_neighbors":

        proto_sim = cls_embeddings.float() @ cls_embeddings.float().t()

        proto_sim.fill_diagonal_(float("-inf"))

        neighbor_topk = max(1, min(neighbor_topk, proto_sim.shape[1] - 1))

        neighbor_idx = torch.topk(proto_sim, k=neighbor_topk, dim=1).indices



        target_neighbor_idx = neighbor_idx[targets]

        candidate_logits = torch.full_like(logits, float("-inf"))

        candidate_logits.scatter_(1, target_neighbor_idx, logits.gather(1, target_neighbor_idx))

    else:

        raise ValueError(f"Unsupported TRAIN.HARD_NEG.POOL: {hard_neg_cfg.POOL}")



    valid_mask = torch.isfinite(candidate_logits)

    valid_counts = valid_mask.sum(dim=1)

    if int((valid_counts > 0).sum().item()) <= 0:

        return None



    cur_topk = min(topk, int(valid_counts.max().item()))

    if cur_topk <= 0:

        return None



    hardest_neg = torch.topk(candidate_logits, k=cur_topk, dim=1).values

    per_neg_loss = F.relu(margin - pos_scores + hardest_neg)

    per_neg_loss = per_neg_loss.masked_fill(~torch.isfinite(hardest_neg), 0.0)



    valid_hard = torch.isfinite(hardest_neg)

    denom = valid_hard.sum().clamp_min(1)

    return per_neg_loss.sum() / denom





def unpack_train_batch(batch):

    if len(batch) == 2:

        captions, targets = batch

        oracle_targets = None

    elif len(batch) == 3:

        captions, targets, oracle_targets = batch

    else:

        raise ValueError(f"Unexpected train batch format with {len(batch)} elements")

    return captions, targets, oracle_targets





def build_oracle_proto_targets(model_module, cls_embeddings, oracle_targets):

    oracle_targets = oracle_targets.float()

    positive_counts = oracle_targets.sum(dim=1, keepdim=True).clamp_min(1.0)

    proto_targets = oracle_targets @ cls_embeddings.float()

    proto_targets = proto_targets / positive_counts

    return model_module._normalize_features(proto_targets).to(dtype=cls_embeddings.dtype)





def get_ot_mode(cfg):

    return getattr(cfg.OT, "MODE", "loss")





def get_eval_topk_list(cfg):

    dataset_name = str(getattr(cfg.DATASET, "NAME", "")).lower()

    if dataset_name == "openimagesv4":

        return [10, 20]

    return [3, 5]





def build_metric_block(predictions, labels, prefix, topk_list=None, dataset_name=None):

    metrics = {}

    if is_openimagesv4_dataset(dataset_name):

        eval_predictions, eval_labels, _ = filter_openimages_biam_classes(predictions, labels, prefix)

        ap, positive_counts, _, _, _ = compute_AP_openimages_biam(

            eval_predictions, eval_labels, return_stats=True

        )

        metrics[f"mAP_{prefix}"] = float(100 * ap.mean().item())

        metrics[f"weighted_mAP_{prefix}"] = compute_weighted_map_percent(ap, positive_counts)

    else:

        ap = compute_AP(predictions, labels)

        metrics[f"mAP_{prefix}"] = float(100 * ap.mean().item())



    if topk_list is None:

        topk_list = [3, 5]



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



    return metrics





def format_metric_block(metrics, prefix, topk_list):

    lines = [f"mAP_{prefix} {metrics[f'mAP_{prefix}']:.2f}"]

    weighted_key = f"weighted_mAP_{prefix}"

    if weighted_key in metrics:

        lines[0] += f" \t weighted_mAP_{prefix} {metrics[weighted_key]:.2f}"

    for k_val in topk_list:

        lines.append(

            "P_{k}_{prefix} {p:.4f} \t R_{k}_{prefix} {r:.4f} \t F1_{k}_{prefix} {f1:.4f}".format(

                k=k_val,

                prefix=prefix,

                p=metrics[f"P_{k_val}_{prefix}"],

                r=metrics[f"R_{k_val}_{prefix}"],

                f1=metrics[f"F1_{k_val}_{prefix}"],

            )

        )

    return "\t \n ".join(lines)





def resolve_align_blend(cfg, epoch, batch_idx=0, num_batches=None):

    base_blend = float(getattr(cfg.OT, "ALIGN_BLEND", 0.0))

    schedule = getattr(cfg.OT, "ALIGN_BLEND_SCHEDULE", "constant")



    if schedule == "constant" or base_blend <= 0:

        return base_blend

    if schedule != "linear_decay":

        raise ValueError(f"Unsupported OT.ALIGN_BLEND_SCHEDULE: {schedule}")



    decay_epochs = int(getattr(cfg.OT, "ALIGN_BLEND_DECAY_EPOCHS", 0) or 0)

    if decay_epochs <= 0:

        raise ValueError("OT.ALIGN_BLEND_DECAY_EPOCHS must be positive when ALIGN_BLEND_SCHEDULE=linear_decay")

    if num_batches is None or num_batches <= 0:

        raise ValueError("num_batches must be positive when ALIGN_BLEND_SCHEDULE=linear_decay")



    total_steps = decay_epochs * num_batches

    if total_steps <= 1:

        return 0.0



    current_step = epoch * num_batches + batch_idx

    current_step = min(max(current_step, 0), total_steps - 1)

    factor = max(0.0, 1.0 - (current_step / float(total_steps - 1)))

    return base_blend * factor





def resolve_epoch_ot_runtime(cfg, epoch, num_batches=None):

    enabled = bool(getattr(cfg.OT, "ENABLED", False))

    runtime = {

        "enabled": enabled,

        "mode": get_ot_mode(cfg),

        "src_mass": getattr(cfg.OT, "SRC_MASS", "uniform"),

        "align_blend": float(getattr(cfg.OT, "ALIGN_BLEND", 0.0)),

    }



    if not enabled:

        return runtime



    runtime["align_blend"] = resolve_align_blend(

        cfg,

        epoch,

        batch_idx=0,

        num_batches=num_batches,

    )

    if num_batches is not None and num_batches > 0:

        runtime["align_blend_end"] = resolve_align_blend(

            cfg,

            epoch,

            batch_idx=num_batches - 1,

            num_batches=num_batches,

        )



    schedule = getattr(cfg.OT, "SRC_MASS_SCHEDULE", "constant")

    switch_epoch = int(getattr(cfg.OT, "SRC_MASS_SWITCH_EPOCH", 0) or 0)

    fallback = getattr(cfg.OT, "SRC_MASS_AFTER", "same")



    if schedule == "constant" or switch_epoch <= 0:

        return runtime

    if schedule != "epoch_switch":

        raise ValueError(f"Unsupported OT.SRC_MASS_SCHEDULE: {schedule}")

    if (epoch + 1) <= switch_epoch:

        return runtime



    if fallback == "uniform":

        runtime["src_mass"] = "uniform"

    elif fallback in {"off", "disable", "disabled"}:

        runtime["enabled"] = False

    elif fallback == "same":

        pass

    else:

        raise ValueError(f"Unsupported OT.SRC_MASS_AFTER: {fallback}")



    return runtime





def compute_ot_loss(model_outputs, cfg, ot_runtime=None):

    caption_seq = model_outputs["caption_seq"]

    target_prompt_parts = model_outputs["target_prompt_parts"]

    caption_mask = model_outputs["caption_mask"]



    if cfg.OT.FP32:

        caption_seq = caption_seq.float()

        target_prompt_parts = target_prompt_parts.float()

        caption_mask = caption_mask.float()



    cost = cosine_cost_matrix(caption_seq, target_prompt_parts)

    src_mass = None

    src_mass_mode = cfg.OT.SRC_MASS if ot_runtime is None else ot_runtime.get("src_mass", cfg.OT.SRC_MASS)

    if src_mass_mode == "uniform":

        pass

    elif src_mass_mode == "ram_softmax":

        sim = 1.0 - cost

        src_mass = build_similarity_source_mass(

            sim=sim,

            src_mask=caption_mask,

            temperature=float(cfg.OT.SRC_MASS_TEMP),

            detach=bool(cfg.OT.SRC_MASS_DETACH),

        )

    else:

        raise ValueError(f"Unsupported OT.SRC_MASS: {src_mass_mode}")



    ot_distance, _ = masked_ot_distance(

        cost=cost,

        src_mask=caption_mask,

        src_mass=src_mass,

        eps=cfg.OT.EPS,

        max_iter=cfg.OT.MAX_ITER,

    )

    return ot_distance.sum()



    

def train(data_loader, model, optim, sched, cfg, logger, epoch):

    batch_time = AverageMeter()

    losses = AverageMeter()

    mse_losses = AverageMeter()

    ot_losses = AverageMeter()

    sim_losses = AverageMeter()

    hard_neg_losses = AverageMeter()

    

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()

    get_model_module(model).text_encoder.eval()

    num_batches = len(data_loader)

    accum_steps = max(1, int(getattr(cfg.TRAIN, "ACCUM_STEPS", 1)))

    ot_runtime = resolve_epoch_ot_runtime(cfg, epoch, num_batches=num_batches)

    ot_mode = ot_runtime["mode"]



    if ot_runtime["enabled"] and ot_mode not in {"loss", "forward_align"}:

        raise ValueError(f"Unsupported OT.MODE: {ot_mode}")

    src_schedule = getattr(cfg.OT, "SRC_MASS_SCHEDULE", "constant")

    blend_schedule = getattr(cfg.OT, "ALIGN_BLEND_SCHEDULE", "constant")

    if cfg.OT.ENABLED and (src_schedule != "constant" or blend_schedule != "constant"):

        log_message = (

            "OT runtime epoch %d: enabled=%s mode=%s src_mass=%s align_blend_start=%.4f"

            % (

                epoch + 1,

                ot_runtime["enabled"],

                ot_mode,

                ot_runtime["src_mass"],

                ot_runtime["align_blend"],

            )

        )

        if "align_blend_end" in ot_runtime:

            log_message += " align_blend_end=%.4f" % ot_runtime["align_blend_end"]

        logger.info(log_message)



    oracle_mode = bool(getattr(cfg.EXP, "ORACLE_MODE", False))

    oracle_proto_losses = AverageMeter()

    oracle_bce_losses = AverageMeter()

    oracle_primary_losses = AverageMeter()



    optim.zero_grad(set_to_none=True)

    end = time.time()

    for i, batch in enumerate(data_loader):

        if torch.cuda.is_available():

            device = torch.device("cuda")

        else:

            device = torch.device("cpu")

        model_outputs = None

        captions, targets, oracle_targets = unpack_train_batch(batch)

        captions = captions.to(device)

        targets = targets.to(device)

        if oracle_targets is not None:

            oracle_targets = oracle_targets.to(device)



        batch_ot_runtime = ot_runtime

        if ot_runtime["enabled"] and blend_schedule != "constant":

            batch_ot_runtime = dict(ot_runtime)

            batch_ot_runtime["align_blend"] = resolve_align_blend(

                cfg,

                epoch,

                batch_idx=i,

                num_batches=num_batches,

            )



        with autocast():

            if ot_runtime["enabled"]:

                model_outputs = model(captions, targets=targets, return_ot=True, ot_runtime=batch_ot_runtime)

                text_features = model_outputs["text_features"]

                cls_embeddings = model_outputs["cls_embeddings"]

            else:

                forward_outputs = model(captions, targets=targets)

                if isinstance(forward_outputs, dict):

                    model_outputs = forward_outputs

                    text_features = model_outputs["text_features"]

                    cls_embeddings = model_outputs["cls_embeddings"]

                else:

                    text_features, cls_embeddings = forward_outputs

        target_indices = model_outputs.get("local_targets", targets) if ot_runtime["enabled"] else targets

        if not ot_runtime["enabled"] and isinstance(model_outputs, dict):

            target_indices = model_outputs.get("local_targets", targets)

        clsname_cur = cls_embeddings[target_indices]

        model_module = get_model_module(model)



        loss_proto = None

        loss_oracle_bce = None

        loss_primary = None



        if oracle_mode and oracle_targets is not None:

            proto_targets = build_oracle_proto_targets(model_module, cls_embeddings, oracle_targets)

            loss_proto = F.mse_loss(proto_targets, text_features, reduction='sum')

            loss_mse = loss_proto

            loss = float(getattr(cfg.EXP, "ORACLE_PROTO_WEIGHT", 1.0)) * loss_proto



            bce_weight = float(getattr(cfg.EXP, "ORACLE_BCE_WEIGHT", 0.0))

            if bce_weight > 0:

                oracle_logits = model_module.logit_scale * (text_features.float() @ cls_embeddings.float().t())

                loss_oracle_bce = F.binary_cross_entropy_with_logits(

                    oracle_logits,

                    oracle_targets.float(),

                    reduction='mean',

                )

                loss = loss + bce_weight * loss_oracle_bce



            primary_weight = float(getattr(cfg.EXP, "ORACLE_PRIMARY_MSE_WEIGHT", 0.0))

            if primary_weight > 0:

                loss_primary = F.mse_loss(clsname_cur, text_features, reduction='sum')

                loss = loss + primary_weight * loss_primary

        else:

            loss_mse = F.mse_loss(clsname_cur, text_features, reduction='sum')                                          

            loss = loss_mse

        loss_ot = None

        loss_sim = None

        loss_hard_neg = None



        if ot_runtime["enabled"] and ot_mode == "loss":

            loss_ot = compute_ot_loss(model_outputs, cfg, ot_runtime=ot_runtime)

            loss = loss + cfg.OT.WEIGHT * loss_ot



        hard_neg_cfg = cfg.TRAIN.HARD_NEG

        hard_neg_enabled = bool(getattr(hard_neg_cfg, "ENABLED", False))

        hard_neg_start_epoch = int(getattr(hard_neg_cfg, "START_EPOCH", 0))

        if hard_neg_enabled and (epoch + 1) >= hard_neg_start_epoch:

            loss_hard_neg = compute_hard_negative_loss(

                text_features=text_features,

                cls_embeddings=cls_embeddings,

                targets=target_indices,

                cfg=cfg,

            )

            if loss_hard_neg is not None:

                hard_neg_weight = float(getattr(hard_neg_cfg, "WEIGHT", 1.0))

                loss = loss + hard_neg_weight * loss_hard_neg



        if cfg.no_sim is True:                                                              

            loss_sim = no_similar_loss(cls_embeddings)

            loss = loss + cfg.no_sim_weights * loss_sim

            

        step_loss = loss / accum_steps

        scaler.scale(step_loss).backward()

        should_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)

        if should_step:

            scaler.step(optim)

            scaler.update()

            optim.zero_grad(set_to_none=True)



        losses.update(loss.item(), captions.size(0))

        mse_losses.update(loss_mse.item(), captions.size(0))

        if loss_proto is not None:

            oracle_proto_losses.update(loss_proto.item(), captions.size(0))

        if loss_oracle_bce is not None:

            oracle_bce_losses.update(loss_oracle_bce.item(), captions.size(0))

        if loss_primary is not None:

            oracle_primary_losses.update(loss_primary.item(), captions.size(0))

        if loss_ot is not None:

            ot_losses.update(loss_ot.item(), captions.size(0))

        if loss_sim is not None:

            sim_losses.update(loss_sim.item(), captions.size(0))

        if loss_hard_neg is not None:

            hard_neg_losses.update(loss_hard_neg.item(), captions.size(0))

        batch_time.update(time.time()-end)

        end = time.time()



        if i % cfg.TRAIN.PRINT_FREQ == 0:

            log_message = (

                'Train: [{0}/{1}]\t'

                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t \t'

                'Loss {losses.val:.2f} ({losses.avg:.2f})\t'

                'Loss_MSE {mse_losses.val:.2f} ({mse_losses.avg:.2f})'

            ).format(

                str(i).rjust(4),

                len(data_loader),

                batch_time=batch_time,

                losses=losses,

                mse_losses=mse_losses,

            )

            if ot_runtime["enabled"] and ot_mode == "loss":

                log_message += '\tLoss_OT {:.2f} ({:.2f})'.format(ot_losses.val, ot_losses.avg)

            if ot_runtime["enabled"] and ot_mode == "forward_align":

                log_message += '\tAlignBlend {:.4f}'.format(batch_ot_runtime["align_blend"])

            if loss_proto is not None:

                log_message += '\tLoss_ORACLE_PROTO {:.2f} ({:.2f})'.format(

                    oracle_proto_losses.val, oracle_proto_losses.avg

                )

            if loss_oracle_bce is not None:

                log_message += '\tLoss_ORACLE_BCE {:.4f} ({:.4f})'.format(

                    oracle_bce_losses.val, oracle_bce_losses.avg

                )

            if loss_primary is not None:

                log_message += '\tLoss_ORACLE_PRIMARY {:.2f} ({:.2f})'.format(

                    oracle_primary_losses.val, oracle_primary_losses.avg

                )

            if cfg.no_sim is True:

                log_message += '\tLoss_SIM {:.2f} ({:.2f})'.format(sim_losses.val, sim_losses.avg)

            if loss_hard_neg is not None:

                log_message += '\tLoss_HNEG {:.4f} ({:.4f})'.format(

                    hard_neg_losses.val, hard_neg_losses.avg

                )

            logger.info(log_message)

    if sched is not None:

        sched.step()

    return batch_time, losses



def validate(data_loader, model, cfg, logger, epoch):

    model.eval()

    preds_unseen = []

    preds_glob = []

    labels_unseen = []

    labels_glob = []

    topk_list = get_eval_topk_list(cfg)



    model_module = get_model_module(model)

    tta_info = get_eval_tta_info(model_module, cfg)

    log_eval_tta_info(logger, tta_info, "Validate")

    rerank_info = get_eval_rerank_info(model_module, cfg)

    log_eval_rerank_info(logger, rerank_info, "Validate")

    chunk_info = get_eval_chunk_info(model_module, cfg)

    log_eval_chunk_info(logger, chunk_info, "Validate")

    num_seen_classes = infer_num_seen_classes(data_loader)

    text_feats = None

    with torch.no_grad():

        if not (tta_info["enabled"] and tta_info["scope"] == "unseen_only" and num_seen_classes is None):

            with autocast():

                text_feats = model_module.encode_text_features(

                    use_tta=tta_info["enabled"],

                    num_seen_classes=num_seen_classes,

                )

    

    for images, seen_labels, unseen_labels in tqdm(data_loader):

        if torch.cuda.is_available():

            device = torch.device("cuda")

        else:

            device = torch.device("cpu")



        images = images.to(device)

        num_seen_cls = seen_labels.size(1)

        if text_feats is None:

            with torch.no_grad():

                with autocast():

                    text_feats = model_module.encode_text_features(

                        use_tta=tta_info["enabled"],

                        num_seen_classes=num_seen_cls,

                    )

        elif num_seen_classes is not None and num_seen_cls != num_seen_classes:

            raise ValueError(

                f"Inconsistent num_seen_classes between dataset ({num_seen_classes}) and batch ({num_seen_cls})"

            )

        

        with torch.no_grad():

            with autocast():

                output = model_module.forward_for_open(images, text_feats, num_seen_classes=num_seen_cls)

                output = model_module.apply_eval_proto_rerank(

                    output, text_feats, num_seen_classes=num_seen_cls

                )

                output = model_module.apply_eval_label_chunking(output, num_seen_classes=num_seen_cls)

            logits_unseen = output[:, num_seen_cls:]

            logits_glob = output

            

        preds_unseen.append(logits_unseen.float().cpu())

        preds_glob.append(logits_glob.float().cpu())

        labels_unseen.append(unseen_labels)

        tmp_labels = torch.cat([seen_labels, unseen_labels], dim=1)

        labels_glob.append(tmp_labels)



    preds_unseen = torch.cat(preds_unseen, dim=0)

    preds_glob = torch.cat(preds_glob, dim=0)

    labels_unseen = torch.cat(labels_unseen, dim=0)

    labels_glob = torch.cat(labels_glob, dim=0)

    logger.info("Evaluating predictions over all images")



    unseen_mask = (labels_unseen > 0).sum(1) > 0

    unseen_metrics = build_metric_block(

        preds_unseen[unseen_mask],

        labels_unseen[unseen_mask],

        "unseen",

        topk_list=topk_list,

        dataset_name=cfg.DATASET.NAME,

    )



    logger.info(

        'Test: [{}/{}] \t {}'.format(

            epoch + 1,

            cfg.OPTIM.MAX_EPOCH,

            format_metric_block(unseen_metrics, "unseen", topk_list),

        )

    )

    

    glob_mask = (labels_glob > 0).sum(1) > 0

    glob_metrics = build_metric_block(

        preds_glob[glob_mask],

        labels_glob[glob_mask],

        "glob",

        topk_list=topk_list,

        dataset_name=cfg.DATASET.NAME,

    )



    logger.info(

        'Test: [{}/{}] \t {}'.format(

            epoch + 1,

            cfg.OPTIM.MAX_EPOCH,

            format_metric_block(glob_metrics, "glob", topk_list),

        )

    )

    

    return {**unseen_metrics, **glob_metrics}



def validate_mssa(data_loader, model, cfg, logger, epoch):

    model.eval()

    preds_unseen = []

    preds_glob = []

    labels_unseen = []

    labels_glob = []

    topk_list = get_eval_topk_list(cfg)



    model_module = get_model_module(model)

    tta_info = get_eval_tta_info(model_module, cfg)

    log_eval_tta_info(logger, tta_info, "Validate")

    rerank_info = get_eval_rerank_info(model_module, cfg)

    log_eval_rerank_info(logger, rerank_info, "Validate")

    chunk_info = get_eval_chunk_info(model_module, cfg)

    log_eval_chunk_info(logger, chunk_info, "Validate")

    num_seen_classes = infer_num_seen_classes(data_loader)

    text_feats = None

    with torch.no_grad():

        if not (tta_info["enabled"] and tta_info["scope"] == "unseen_only" and num_seen_classes is None):

            with autocast():

                text_feats = model_module.encode_text_features(

                    use_tta=tta_info["enabled"],

                    num_seen_classes=num_seen_classes,

                )

    

    for images, seen_labels, unseen_labels, img_list in tqdm(data_loader):

        if torch.cuda.is_available():

            device = torch.device("cuda")

        else:

            device = torch.device("cpu")



        images = images.to(device)

        num_seen_cls = seen_labels.size(1)

        if text_feats is None:

            with torch.no_grad():

                with autocast():

                    text_feats = model_module.encode_text_features(

                        use_tta=tta_info["enabled"],

                        num_seen_classes=num_seen_cls,

                    )

        elif num_seen_classes is not None and num_seen_cls != num_seen_classes:

            raise ValueError(

                f"Inconsistent num_seen_classes between dataset ({num_seen_classes}) and batch ({num_seen_cls})"

            )

        

        output_list = []

        with torch.no_grad():

            with autocast():

                output = model_module.forward_for_test(images, text_feats, num_seen_classes=num_seen_cls)

                for img in img_list:

                    output_ = model_module.forward_for_test(img.to(device), text_feats, num_seen_classes=num_seen_cls)

                    output_list.append(output_)

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

        tmp_labels = torch.cat([seen_labels, unseen_labels], dim=1)

        labels_glob.append(tmp_labels)



    preds_unseen = torch.cat(preds_unseen, dim=0)

    preds_glob = torch.cat(preds_glob, dim=0)

    labels_unseen = torch.cat(labels_unseen, dim=0)

    labels_glob = torch.cat(labels_glob, dim=0)

    logger.info("Evaluating predictions over all images")



    unseen_mask = (labels_unseen > 0).sum(1) > 0

    unseen_metrics = build_metric_block(

        preds_unseen[unseen_mask],

        labels_unseen[unseen_mask],

        "unseen",

        topk_list=topk_list,

        dataset_name=cfg.DATASET.NAME,

    )



    logger.info(

        'Test: [{}/{}] \t {}'.format(

            epoch + 1,

            cfg.OPTIM.MAX_EPOCH,

            format_metric_block(unseen_metrics, "unseen", topk_list),

        )

    )

    

    glob_mask = (labels_glob > 0).sum(1) > 0

    glob_metrics = build_metric_block(

        preds_glob[glob_mask],

        labels_glob[glob_mask],

        "glob",

        topk_list=topk_list,

        dataset_name=cfg.DATASET.NAME,

    )



    logger.info(

        'Test: [{}/{}] \t {}'.format(

            epoch + 1,

            cfg.OPTIM.MAX_EPOCH,

            format_metric_block(glob_metrics, "glob", topk_list),

        )

    )

    

    return {**unseen_metrics, **glob_metrics}

