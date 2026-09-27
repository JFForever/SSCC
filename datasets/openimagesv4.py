import csv

import json

from pathlib import Path



import numpy as np

import torch

from PIL import Image

from torch.utils.data import Dataset





def _read_lines(path):

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fp:

        return [line.strip() for line in fp if line.strip()]





def _load_json(path):

    with open(path, "r", encoding="utf-8") as fp:

        return json.load(fp)





def _load_label_names(csv_path):

    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as fp:

        rows = list(csv.reader(fp))

    if not rows:

        return []

    header = rows[0]

    if len(header) >= 2 and header[0] == "LabelName":

        return [(row[0].strip(), row[1].strip()) for row in rows[1:] if len(row) >= 2]

    return [(row[0].strip(), row[1].strip()) for row in rows if len(row) >= 2]





def _normalize_display_name(name):

    return str(name).strip().replace("_", " ").replace("-", " ")





def _resolve_dataset_root(anno_path):

    path = Path(anno_path).expanduser().resolve()

    if (path / "split").is_dir() and (path / "processed").is_dir():

        return path



    if path.name == "annotations" and (path.parent / "split").is_dir():

        return path.parent



    if (path.parent / "split").is_dir() and (path.parent / "processed").is_dir():

        return path.parent



    return path





def _load_split_manifest(root):

    manifest_path = root / "processed" / "dataset_manifest_7186_seen_400_unseen.json"

    if not manifest_path.is_file():

        return None

    try:

        return _load_json(manifest_path)

    except Exception:

        return None





def _load_split_labels(root):

    split_dir = root / "split"

    seen_ids = _read_lines(split_dir / "seen_7186_label_ids.txt")

    unseen_ids = _read_lines(split_dir / "unseen_400_label_ids.txt")

    seen_names = _load_label_names(split_dir / "seen_7186_label_names.csv")

    unseen_names = _load_label_names(split_dir / "unseen_400_label_names.csv")

    if len(seen_ids) != 7186 or len(unseen_ids) != 400:

        raise ValueError(

            f"OpenImagesV4 split mismatch: seen={len(seen_ids)} unseen={len(unseen_ids)}"

        )

    return seen_ids, unseen_ids, seen_names, unseen_names





class OpenImagesV4(Dataset):

    def __init__(self, cfg, img_path, anno_path, data_set="train", transform=None):

        super().__init__()

        self.cfg = cfg

        self.data_set = data_set

        self.transform = transform



        self.dataset_root = _resolve_dataset_root(anno_path)

        self.img_root = Path(img_path).expanduser().resolve()

        self.annotations_dir = self.dataset_root / "annotations"

        self.metadata_dir = self.dataset_root / "metadata"

        self.image_lists_dir = self.dataset_root / "image_lists"

        self.processed_dir = self.dataset_root / "processed"

        self.split_dir = self.dataset_root / "split"



        required_dirs = {

            "dataset root": self.dataset_root,

            "annotations": self.annotations_dir,

            "metadata": self.metadata_dir,

            "image_lists": self.image_lists_dir,

            "processed": self.processed_dir,

            "split": self.split_dir,

            "image root": self.img_root,

        }

        for label, directory in required_dirs.items():

            if not directory.is_dir():

                raise FileNotFoundError(f"OpenImagesV4 {label} directory not found: {directory}")



        self.manifest = _load_split_manifest(self.dataset_root)

        self.seen_label_ids, self.unseen_label_ids, self.seen_name_rows, self.unseen_name_rows = _load_split_labels(

            self.dataset_root

        )

        self.seen_names = np.array([_normalize_display_name(name) for _, name in self.seen_name_rows])

        self.unseen_names = np.array([_normalize_display_name(name) for _, name in self.unseen_name_rows])

        self.classnames = list(self.seen_names) + list(self.unseen_names)



        self.seen_label_index = {label_id: idx for idx, label_id in enumerate(self.seen_label_ids)}

        self.unseen_label_index = {label_id: idx for idx, label_id in enumerate(self.unseen_label_ids)}

        self.label_to_global_index = {

            label_id: idx for idx, label_id in enumerate(list(self.seen_label_ids) + list(self.unseen_label_ids))

        }



        self.labels_by_image = self._load_labels_for_split(data_set)

        self.img_names = sorted(self.labels_by_image.keys())



        self.seen_cls_idx = list(range(len(self.seen_names)))

        self.unseen_cls_idx = list(range(len(self.seen_names), len(self.classnames)))



        if data_set == "train":

            self.label_dim = len(self.seen_names)

        else:

            self.label_dim = len(self.classnames)



    def _load_labels_for_split(self, data_set):

        split_name = data_set.lower()

        if split_name == "train":

            ann_file = self.processed_dir / "train_seen_7186_annotations.csv"

            list_file = self.image_lists_dir / "train_image_ids.txt"

        elif split_name in {"val", "validation"}:

            ann_file = self.processed_dir / "validation_seen_7186_annotations.csv"

            list_file = self.image_lists_dir / "validation_image_ids.txt"

        elif split_name == "test":

            ann_file = self.processed_dir / "test_seen_7186_unseen_400_annotations.csv"

            list_file = self.image_lists_dir / "test_gzsl_image_ids.txt"

        else:

            raise ValueError(f"Unsupported OpenImagesV4 split: {data_set}")



        if not ann_file.is_file():

            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        if not list_file.is_file():

            raise FileNotFoundError(f"Image id list not found: {list_file}")



        allowed_ids = set(_read_lines(list_file))



        labels_by_image = {}

        with open(ann_file, "r", encoding="utf-8-sig", errors="ignore") as fp:

            reader = csv.DictReader(fp)

            for row in reader:

                image_id = row["ImageID"].strip()

                if image_id not in allowed_ids:

                    continue

                label_id = row["LabelName"].strip()

                confidence = row["Confidence"].strip()

                labels_by_image.setdefault(image_id, {})

                labels_by_image[image_id][label_id] = 1 if confidence == "1" else 0



        if not labels_by_image:

            raise RuntimeError(f"No labels loaded for OpenImagesV4 split {data_set}")



        return labels_by_image



    def _build_target_tensor(self, image_id):

        if self.data_set == "train":

            target = torch.zeros(len(self.seen_names), dtype=torch.long)

            for label_id, confidence in self.labels_by_image[image_id].items():

                if label_id in self.seen_label_index:

                    target[self.seen_label_index[label_id]] = 1 if confidence == 1 else -1

            return target



        target = torch.zeros(len(self.classnames), dtype=torch.long)

        for label_id, confidence in self.labels_by_image[image_id].items():

            global_idx = self.label_to_global_index.get(label_id)

            if global_idx is None:

                continue

            target[global_idx] = 1 if confidence == 1 else -1

        return target



    def __len__(self):

        return len(self.img_names)



    def __getitem__(self, index):

        image_id = self.img_names[index]

        image_path = self._resolve_image_path(image_id)

        if image_path is None:

            raise FileNotFoundError(f"Image file not found for {image_id} under {self.img_root}")



        img = Image.open(image_path).convert("RGB")

        if self.transform is not None:

            img_ = self.transform(img)

        else:

            img_ = img



        target = self._build_target_tensor(image_id)

        if self.data_set == "train":

            seen_label = target

            return img_, seen_label



                                                                                   

        seen_label = torch.zeros(len(self.seen_names), dtype=torch.long)

        unseen_label = torch.zeros(len(self.unseen_names), dtype=torch.long)

        for label_id, confidence in self.labels_by_image[image_id].items():

            if label_id in self.seen_label_index:

                seen_label[self.seen_label_index[label_id]] = 1 if confidence == 1 else -1

            elif label_id in self.unseen_label_index:

                unseen_label[self.unseen_label_index[label_id]] = 1 if confidence == 1 else -1



        if self.cfg.cutimage:

            img_list = CutImageplus(img, count_list=self.cfg.count_list)

            img_list = [self.transform(image) for image in img_list]

            return img_, seen_label, unseen_label, img_list

        return img_, seen_label, unseen_label



    def _resolve_image_path(self, image_id):

        split = self.data_set.lower()

        direct_candidates = [

            self.img_root / f"{image_id}.jpg",

            self.img_root / split / f"{image_id}.jpg",

            self.img_root / split / split / f"{image_id}.jpg",

            self.img_root / split / split.upper() / f"{image_id}.jpg",

        ]

        if split == "train":

            direct_candidates.extend(

                [

                    self.img_root / "train" / f"train_{image_id[0]}" / f"{image_id}.jpg",

                    self.img_root / "train" / "train" / f"{image_id}.jpg",

                ]

            )

        else:

            direct_candidates.append(self.img_root / split / split / f"{image_id}.jpg")



        for candidate in direct_candidates:

            if candidate.is_file():

                return candidate



        split_root = self.img_root / split

        if split_root.is_dir():

            matches = list(split_root.rglob(f"{image_id}.jpg"))

            if matches:

                return matches[0]

        return None





def CutImageplus(img, img_size=224, count_list=[2, 3]):

    img = img.resize((img_size, img_size))

    box_list = []

    for count in count_list:

        temp_width = int(img_size / count)

        for i in range(0, count):

            for j in range(0, count):

                box = (j * temp_width, i * temp_width, (j + 1) * temp_width, (i + 1) * temp_width)

                box_list.append(box)

    return [img.crop(box) for box in box_list]

