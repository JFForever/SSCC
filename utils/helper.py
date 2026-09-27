import os

import torch

import random

import codecs

import logging

import shutil

import numpy as np





def setup_seed(seed):

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed) 

    random.seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True





def init_log(record_path):

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    log_path = os.path.join(record_path, 'recording.log')



    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

    

    fh = logging.FileHandler(log_path, mode='w') 

    fh.setLevel(logging.DEBUG)

    fh.setFormatter(formatter)

    logger.addHandler(fh)



    sh = logging.StreamHandler() 

    sh.setLevel(logging.DEBUG)

    sh.setFormatter(formatter)

    logger.addHandler(sh)

    

    return logger





def write_description_to_folder(file_name, config):

    with codecs.open(file_name, 'w') as desc_f:

        desc_f.write("- Training Parameters: \n")

        desc_f.write(str(config))





def convert_models_to_fp32(model):

    for p in model.parameters():

        p.data = p.data.float()

        if p.grad:

            p.grad.data = p.grad.data.float()





def convert_models_to_half(model):

    for p in model.parameters():

        p.data = p.data.half()

        if p.grad:

            p.grad.data = p.grad.data.half()





def compute_AP(predictions, labels):

    num_class = predictions.size(1)

    ap = torch.zeros(num_class).to(predictions.device)

    empty_class = 0

    for idx_cls in range(num_class):

        prediction = predictions[:, idx_cls]

        label = labels[:, idx_cls]

        if (label > 0).sum() == 0:

            empty_class += 1

            continue



        if (label == -1).sum() != 0:

            mask = label.abs() == 1

            label = torch.clamp(label[mask], min=0, max=1)

            prediction = prediction[mask]  

        sorted_pred, sort_idx = prediction.sort(descending=True)

        sorted_label = label[sort_idx]

        tmp = (sorted_label == 1).float()

        tp = tmp.cumsum(0)

        fp = (sorted_label != 1).float().cumsum(0)

        num_pos = label.sum()

        rec = tp/num_pos

        prec = tp/(tp+fp)

        ap_cls = (tmp*prec).sum()/num_pos

        ap[idx_cls].copy_(ap_cls)

    return ap





def is_openimagesv4_dataset(dataset_name):

    return str(dataset_name).lower() == "openimagesv4"





def filter_openimages_biam_classes(predictions, labels, prefix):

    if prefix != "glob":

        class_mask = torch.ones(labels.size(1), dtype=torch.bool, device=labels.device)

        return predictions, labels, class_mask



    class_mask = (labels != 0).sum(dim=0) > 0

    return predictions[:, class_mask], labels[:, class_mask], class_mask





def compute_AP_openimages_biam(predictions, labels, return_stats=False):

    num_class = predictions.size(1)

    ap = torch.zeros(num_class, dtype=torch.float32, device=predictions.device)

    positive_counts = torch.zeros(num_class, dtype=torch.long, device=labels.device)

    negative_counts = torch.zeros(num_class, dtype=torch.long, device=labels.device)

    ignored_counts = torch.zeros(num_class, dtype=torch.long, device=labels.device)

    valid_counts = torch.zeros(num_class, dtype=torch.long, device=labels.device)



    for idx_cls in range(num_class):

        prediction = predictions[:, idx_cls]

        label = labels[:, idx_cls]



        valid_mask = label != 0

        ignored_counts[idx_cls] = (~valid_mask).sum()

        prediction = prediction[valid_mask]

        label = label[valid_mask]



        valid_counts[idx_cls] = label.numel()

        if label.numel() == 0:

            continue



        binary_label = torch.clamp(label, min=0, max=1)

        positive_counts[idx_cls] = (binary_label > 0).sum()

        negative_counts[idx_cls] = valid_counts[idx_cls] - positive_counts[idx_cls]



        if positive_counts[idx_cls].item() <= 0:

            continue



        sorted_pred, sort_idx = prediction.sort(descending=True)

        sorted_label = binary_label[sort_idx]

        tmp = (sorted_label == 1).float()

        tp = tmp.cumsum(0)

        fp = (sorted_label != 1).float().cumsum(0)

        num_pos = binary_label.sum()

        prec = tp / (tp + fp)

        ap_cls = (tmp * prec).sum() / num_pos

        ap[idx_cls].copy_(ap_cls)



    if return_stats:

        return ap, positive_counts, negative_counts, ignored_counts, valid_counts

    return ap





def compute_weighted_map_percent(ap, positive_counts):

    weights = positive_counts.float()

    total = weights.sum()

    if total.item() <= 0:

        return 0.0

    return float((100.0 * (ap * weights).sum() / total).item())





def compute_F1(predictions, labels, k_val):

    idx = predictions.topk(dim=1, k=k_val)[1]

    predictions.fill_(0)

    predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val).to(predictions.device))

    mask = predictions == 1

    TP = (labels[mask] == 1).sum().float()

    tpfp = mask.sum().float()

    tpfn = (labels == 1).sum().float()

    p = TP / tpfp

    r = TP/tpfn

    f1 = 2*p*r/(p+r)



    return f1, p, r





class AverageMeter(object):

                                                           

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count





def save_checkpoint(state, is_best, filepath='', prefix=None):

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))

    if is_best:

        if prefix is None:

            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

        else:

            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, '%s_model_best.pth.tar' % prefix))

