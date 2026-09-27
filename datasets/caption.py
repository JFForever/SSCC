import os

import json

import math

import pickle

import numpy as np

import torch

from clip import clip

from torch.utils.data import Dataset



imagenet_templates = [

    'a bad photo of a {}.',

    'a photo of many {}.',

    'a sculpture of a {}.',

    'a photo of the hard to see {}.',

    'a low resolution photo of the {}.',

    'a rendering of a {}.',

    'graffiti of a {}.',

    'a bad photo of the {}.',

    'a cropped photo of the {}.',

    'a tattoo of a {}.',

    'the embroidered {}.',

    'a photo of a hard to see {}.',

    'a bright photo of a {}.',

    'a photo of a clean {}.',

    'a photo of a dirty {}.',

    'a dark photo of the {}.',

    'a drawing of a {}.',

    'a photo of my {}.',

    'the plastic {}.',

    'a photo of the cool {}.',

    'a close-up photo of a {}.',

    'a black and white photo of the {}.',

    'a painting of the {}.',

    'a painting of a {}.',

    'a pixelated photo of the {}.',

    'a sculpture of the {}.',

    'a bright photo of the {}.',

    'a cropped photo of a {}.',

    'a plastic {}.',

    'a photo of the dirty {}.',

    'a jpeg corrupted photo of a {}.',

    'a blurry photo of the {}.',

    'a photo of the {}.',

    'a good photo of the {}.',

    'a rendering of the {}.',

    'a {} in a video game.',

    'a photo of one {}.',

    'a doodle of a {}.',

    'a close-up photo of the {}.',

    'a photo of a {}.',

    'the origami {}.',

    'the {} in a video game.',

    'a sketch of a {}.',

    'a doodle of the {}.',

    'a origami {}.',

    'a low resolution photo of a {}.',

    'the toy {}.',

    'a rendition of the {}.',

    'a photo of the clean {}.',

    'a photo of a large {}.',

    'a rendition of a {}.',

    'a photo of a nice {}.',

    'a photo of a weird {}.',

    'a blurry photo of a {}.',

    'a cartoon {}.',

    'art of a {}.',

    'a sketch of the {}.',

    'a embroidered {}.',

    'a pixelated photo of a {}.',

    'itap of the {}.',

    'a jpeg corrupted photo of the {}.',

    'a good photo of a {}.',

    'a plushie {}.',

    'a photo of the nice {}.',

    'a photo of the small {}.',

    'a photo of the weird {}.',

    'the cartoon {}.',

    'art of the {}.',

    'a drawing of the {}.',

    'a photo of the large {}.',

    'a black and white photo of a {}.',

    'the plushie {}.',

    'a dark photo of a {}.',

    'itap of a {}.',

    'graffiti of the {}.',

    'a toy {}.',

    'itap of my {}.',

    'a photo of a cool {}.',

    'a photo of a small {}.',

    'a tattoo of the {}.',

]



def load_json(filename: str):

    if not filename.endswith('.json'):

        filename += '.json'

    with open(filename, 'r') as fp:

        return json.load(fp)





def _normalize_class_name(name):

    return str(name).strip().replace('_', ' ').lower()



class Caption(Dataset):

    def __init__(self, text_path, dataset='mscoco', extra_template=False, cfg=None, classnames=None):

        super(Caption, self).__init__()

        self.cfg = cfg

        self.dataset = dataset.lower()

        self.oracle_mode = bool(cfg is not None and getattr(cfg.EXP, "ORACLE_MODE", False))

        self.classnames = list(classnames) if classnames is not None else None

        self.num_classes = len(self.classnames) if self.classnames is not None else 0

        text_json_name = ""

        if cfg is not None:

            text_json_name = str(getattr(cfg.DATASET, "TEXT_JSON", "") or "").strip()

        if text_json_name:

            if os.path.isabs(text_json_name):

                text_path = text_json_name

            else:

                text_path = os.path.join(text_path, text_json_name)

        else:

            text_path = os.path.join(text_path, '{}_post_llm.json'.format(dataset.lower()))

        self.text_data = load_json(text_path)

        self.extra_caption_map = self._load_extra_caption_map()

        self.train_class_filter = self._load_train_class_filter()

        self.extra_template = bool(extra_template and cfg is not None and getattr(cfg.TRAINER.TEXT, "EXTRA_TEMPLATE", True))



        base_train = []

        text_data = self.text_data.keys()



        for ind, idx in enumerate(text_data):

            cls_name = self.text_data[idx]['node_name']

            cls_name = cls_name.replace('_', ' ')

            cls_name_norm = _normalize_class_name(cls_name)

            if self.train_class_filter and cls_name_norm not in self.train_class_filter:

                continue

            sentences = self.text_data[idx]['candidate_sentences']

            for sentence in sentences:

                sentence_p = clip.tokenize(sentence)[0]

                target = int(idx)

                item_ = (sentence_p, target)

                base_train.append(item_)



            extra_sentences = self.extra_caption_map.get(cls_name_norm, [])

            for sentence in extra_sentences:

                sentence_p = clip.tokenize(sentence)[0]

                target = int(idx)

                base_train.append((sentence_p, target))



            if self.extra_template:

                for cur_temp in imagenet_templates:

                    temp_p = clip.tokenize(cur_temp.format(cls_name))[0]

                    target = int(idx)

                    base_train.append((temp_p, target))



        if self.oracle_mode:

            if self.dataset != 'nuswide':

                raise NotImplementedError("Oracle mode is currently implemented for nuswide only")

            self.train = self._build_nuswide_oracle_samples()

            if bool(getattr(self.cfg.EXP, "ORACLE_INCLUDE_BASE_CAPTIONS", False)):

                self.train.extend(

                    (prompt, target, [target])

                    for prompt, target in base_train

                )

        else:

            self.train = base_train



    def _load_extra_caption_map(self):

        if self.cfg is None:

            return {}



        extra_path = str(getattr(self.cfg.EXP, "EXTRA_CAPTION_JSON", "") or "").strip()

        if not extra_path:

            return {}



        if not os.path.isabs(extra_path):

            extra_path = os.path.join(self.cfg.DATASET.TEXT_PATH, extra_path)

        if not os.path.exists(extra_path):

            raise FileNotFoundError(f"Extra caption json not found: {extra_path}")



        payload = load_json(extra_path)

        if not isinstance(payload, dict):

            raise ValueError("Extra caption json must be a dict mapping class names to sentence lists")



        extra_caption_map = {}

        for raw_name, raw_sentences in payload.items():

            class_name = _normalize_class_name(raw_name)

            if not isinstance(raw_sentences, list):

                raise ValueError(f"Extra caption entry for '{raw_name}' must be a list of strings")



            cleaned = []

            for sentence in raw_sentences:

                text = str(sentence).strip()

                if not text:

                    continue

                cleaned.append(text)

            if cleaned:

                extra_caption_map[class_name] = cleaned



        return extra_caption_map



    def _load_train_class_filter(self):

        if self.cfg is None:

            return set()



        filter_path = str(getattr(self.cfg.EXP, "TRAIN_CLASS_FILTER_JSON", "") or "").strip()

        if not filter_path:

            return set()



        if not os.path.isabs(filter_path):

            filter_path = os.path.join(self.cfg.DATASET.TEXT_PATH, filter_path)

        if not os.path.exists(filter_path):

            raise FileNotFoundError(f"Train class filter json not found: {filter_path}")



        payload = load_json(filter_path)

        if isinstance(payload, dict):

            class_names = payload.keys()

        elif isinstance(payload, list):

            class_names = payload

        else:

            raise ValueError("Train class filter json must be a dict or list of class names")



        filtered = {_normalize_class_name(name) for name in class_names if str(name).strip()}

        if not filtered:

            raise ValueError("Train class filter json resolved to an empty class set")

        return filtered



    def _build_nuswide_oracle_samples(self):

        from .nuswide import load_pickle



        if not self.classnames:

            raise ValueError("classnames must be provided when EXP.ORACLE_MODE=True")



        anno_path = self.cfg.DATASET.ANNO_PATH

        train_label_path = os.path.join(anno_path, 'train_seen_labels.pkl')

        if not os.path.exists(train_label_path):

            self._build_nuswide_train_seen_labels(anno_path, train_label_path)



        with open(os.path.join(anno_path, 'TagList1k.txt'), 'r') as fp:

            raw_classnames = [line.strip().replace('_', ' ') for line in fp if line.strip()]



        final_name_to_idx = {

            cls_name.replace('_', ' '): idx

            for idx, cls_name in enumerate(self.classnames)

        }

        raw_to_final = []

        for cls_name in raw_classnames:

            if cls_name not in final_name_to_idx:

                raise KeyError(f"Class '{cls_name}' from TagList1k.txt was not found in final classnames")

            raw_to_final.append(final_name_to_idx[cls_name])



        label_dict = load_pickle(train_label_path)

        oracle_text_mode = str(getattr(self.cfg.EXP, "ORACLE_TEXT_MODE", "multilabel_sentence")).lower()

        template = str(getattr(self.cfg.EXP, "ORACLE_TEXT_TEMPLATE", "a photo containing {}."))

        max_labels = int(getattr(self.cfg.EXP, "ORACLE_MAX_LABELS_PER_SAMPLE", 0) or 0)

        sample_limit = int(getattr(self.cfg.EXP, "ORACLE_SAMPLE_LIMIT", 0) or 0)



        train = []

        for img_name, raw_labels in label_dict.items():

            raw_labels = list(raw_labels)

            positive_raw_indices = [idx for idx, value in enumerate(raw_labels) if value > 0]

            if len(positive_raw_indices) <= 0:

                continue



            positive_final = [raw_to_final[idx] for idx in positive_raw_indices]

            positive_names = [self.classnames[idx].replace('_', ' ') for idx in positive_final]



            if oracle_text_mode == "single_label":

                for final_idx, cls_name in zip(positive_final, positive_names):

                    prompt = self._tokenize_oracle_text(template.format(self._format_single_label_name(cls_name)))

                    train.append((prompt, final_idx, [final_idx]))

            elif oracle_text_mode == "multilabel_sentence":

                chunk_size = max_labels if max_labels > 0 else len(positive_final)

                for start in range(0, len(positive_final), chunk_size):

                    chunk_indices = positive_final[start:start + chunk_size]

                    chunk_names = positive_names[start:start + chunk_size]

                    prompt = self._tokenize_oracle_text(template.format(self._format_multilabel_phrase(chunk_names)))

                    train.append((prompt, chunk_indices[0], chunk_indices))

            else:

                raise ValueError(f"Unsupported EXP.ORACLE_TEXT_MODE: {oracle_text_mode}")



            if sample_limit > 0 and len(train) >= sample_limit:

                train = train[:sample_limit]

                break



        if len(train) <= 0:

            raise RuntimeError("Oracle dataset produced zero samples")



        return train



    def _build_nuswide_train_seen_labels(self, anno_path, output_path):

        imagelist_path = os.path.join(anno_path, 'Imagelist.txt')

        testlist_path = os.path.join(anno_path, 'TestImagelist.txt')

        alltags_path = os.path.join(anno_path, 'AllTags1k.txt')



        with open(imagelist_path, 'r') as fp:

            image_ids = [line.strip().replace('\\', '/') for line in fp if line.strip()]

        with open(testlist_path, 'r') as fp:

            test_ids = {line.strip().replace('\\', '/') for line in fp if line.strip()}

        with open(alltags_path, 'r') as fp:

            all_label_rows = [

                np.array(line.strip().split('\t'), dtype=np.float32)

                for line in fp if line.strip()

            ]



        if len(image_ids) != len(all_label_rows):

            raise ValueError(

                f"Imagelist and AllTags1k length mismatch: {len(image_ids)} vs {len(all_label_rows)}"

            )



        train_seen_dict = {}

        for image_id, label_row in zip(image_ids, all_label_rows):

            if image_id in test_ids:

                continue

            train_seen_dict[image_id] = 2 * label_row - 1



        if len(train_seen_dict) <= 0:

            raise RuntimeError("Recovered zero train samples for NUSWIDE oracle split")



        with open(output_path, 'wb') as fp:

            pickle.dump(train_seen_dict, fp)



    def _format_single_label_name(self, cls_name):

        vowels = {"a", "e", "i", "o", "u"}

        article = "an" if cls_name[:1].lower() in vowels else "a"

        return f"{article} {cls_name}"



    def _format_multilabel_phrase(self, class_names):

        if len(class_names) == 1:

            return self._format_single_label_name(class_names[0])

        if len(class_names) == 2:

            return f"{class_names[0]} and {class_names[1]}"

        return ", ".join(class_names[:-1]) + f", and {class_names[-1]}"



    def _tokenize_oracle_text(self, text):

        try:

            return clip.tokenize(text)[0]

        except RuntimeError:

            class_names = [part.strip() for part in text.replace("a photo containing", "").strip(" .").split(",")]

            if len(class_names) <= 1:

                raise

            keep = max(1, math.ceil(len(class_names) / 2))

            shorter_text = str(getattr(self.cfg.EXP, "ORACLE_TEXT_TEMPLATE", "a photo containing {}.")).format(

                self._format_multilabel_phrase(class_names[:keep])

            )

            return clip.tokenize(shorter_text)[0]



    def __getitem__(self, index):

        if self.oracle_mode:

            prompt, label, pos_indices = self.train[index]

            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)

            multi_hot[pos_indices] = 1.0

            return prompt, label, multi_hot



        prompt, label = self.train[index]

        return prompt, label

    

    def __len__(self):

        return len(self.train)

