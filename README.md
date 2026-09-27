# SSCC

Official implementation of **SSCC** for multi-label visual recognition.

## Datasets
We conduct experiments on three public benchmark datasets: MSCOCO, NUSWIDE, and OpenImage V4.

- Download images of MSCOCO dataset from [MSCOCO](https://cocodataset.org/#download).
- Download images of NUS-WIDE dataset from [NUSWIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html).
- Download images of OpenImage V4 dataset from [OpenImage V4](https://storage.googleapis.com/openimages/web/index.html).

## Environment Requirements
- Python == 3.10
- PyTorch == 2.1.0
- Other dependencies are listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt


SSCC
├── clip/             # CLIP backbone modules
├── datasets/         # Dataset loading and preprocessing
├── model/            # Core SSCC model architecture
├── utils/            # Helper functions and tools
├── train_text.py     # Training pipeline
├── eval_checkpoint.py# Evaluation pipeline
└── requirements.txt  # Environment configuration



## Acknowledgements

We sincerely appreciate the excellent open-source projects from [RCNn](https://github.com/wangshouwen/RCNn/tree/main), [CoOp](https://github.com/KaiyangZhou/CoOp) and [MKT](https://github.com/sunanhe/MKT). Our code is heavily built on their official implementations.
