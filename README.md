# Adapting with an Open Mind: Leveraging Open-Vocabulary Detectors for Closed Set Source-Free Domain Adaptive Object Detection

*Kaustubh R Borgavi, Sarvesh Shashikumar, Chetan Arora*

Official PyTorch implementation of `Adapting with an Open Mind: Leveraging Open-Vocabulary Detectors for Closed Set Source-Free Domain Adaptive Object Detection` [CVPR 2026 (Findings)]

<div align="center">

## 🚧 Code Release In Progress 🚧

**Note:** Code release is underway! We are currently cleaning up the codebase and performing final checks at our end. Please check back soon for the complete release. If you'd like to stay updated, consider ⭐ starring the repository. Thank you, and we hope you find our work useful!

</div>

<p align="center">
  <img width="75%" alt="method_overview" src="adapting-with-an-open-mind.png">
</p>

## 🔔 Updates

- **[Jun 2026]** Code and pretrained weights are released.
- **[Jun 2026]** Paper presented at CVPR 2026.

## ✅ ToDo

- [x] Release training and evaluation scripts
- [ ] Release pretrained model weights
- [ ] Revert scripts to relative paths wherever needed.
- [ ] Add Def-DeTR codebase.
- [ ] Add G-DINO and BMP integration setup with instructions in README. 

## 1. Installation

### 1.1 Requirements
- Linux, CUDA >= 11.1, GCC >= 8.4
- Python >= 3.8
- torch >= 1.10.1, torchvision >= 0.11.2
- Other requirements
```bash
  pip install -r requirements.txt
```

### 1.2 Compiling CUDA Operators (if applicable)
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## 2. Dataset Preparation

Our method is evaluated on 3 popular SFOD benchmarks:
- **city2foggy**: Cityscapes (source) → FoggyCityscapes with foggy level 0.02 (target)
- **sim2city**: Sim10k (source) → Cityscapes with `car` class (target)
- **city2bdd**: Cityscapes (source) → Bdd100k-daytime (target)

Download the raw data from the official websites:
[Cityscapes](https://www.cityscapes-dataset.com/downloads/) |
[FoggyCityscapes](https://www.cityscapes-dataset.com/downloads/) |
[Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) |
[Bdd100k](https://bdd-data.berkeley.edu/)

Annotations are in COCO style and can be downloaded from [here](#) (provided by [MRT-release](https://github.com/JeremyZhao1998/MRT-release)).

Organize datasets and annotations as follows:

```bash
[data_root]
└─ cityscapes
    └─ annotations
        └─ cityscapes_train_cocostyle.json
        └─ cityscapes_train_caronly_cocostyle.json
        └─ cityscapes_val_cocostyle.json
        └─ cityscapes_val_caronly_cocostyle.json
    └─ leftImg8bit
        └─ train
        └─ val
└─ foggy_cityscapes
    └─ annotations
        └─ foggy_cityscapes_train_cocostyle.json
        └─ foggy_cityscapes_val_cocostyle.json
    └─ leftImg8bit_foggy
        └─ train
        └─ val
└─ sim10k
    └─ annotations
        └─ sim10k_train_cocostyle.json
    └─ JPEGImages
└─ bdd10k
    └─ annotations
        └─ bdd100k_daytime_train_cocostyle.json
        └─ bdd100k_daytime_val_cocostyle.json
    └─ images
```

## 3. Training and Evaluation

### 3.1 Training

First, run `source_only` to pretrain the source-only model. Then run `teaching_standard` to train the baseline or `teaching_ours` to train our proposed method.

For example, for the `city2foggy` benchmark, first edit the files in `configs/def-detr-base/city2foggy/` to specify your `DATA_ROOT` and `OUTPUT_DIR`, then run:

```bash
sh configs/def-detr-base/city2foggy/source_only.sh
sh configs/def-detr-base/city2foggy/teaching_standard.sh
sh configs/def-detr-base/city2foggy/teaching_ours.sh
```

### 3.2 Evaluation

```bash
sh configs/def-detr-base/city2foggy/evaluation_source_only.sh
sh configs/def-detr-base/city2foggy/evaluation_teaching_standard.sh
sh configs/def-detr-base/city2foggy/evaluation_teaching_ours.sh
```

## 4. Experiments

All experiments are conducted with batch size 8, on an NVIDIA [GPU] ([X]GB).

**city2foggy**: Cityscapes → FoggyCityscapes (level 0.02)

| Training Stage      | AP@50 | Logs & Weights |
| ------------------- | ----- | -------------- |
| `source_only`       | -     | [Source-only](#)|
| `teaching_standard` | -     | [Baseline](#)  |
| `teaching_ours`     | -     | [Ours](#)      |

**city2bdd**: Cityscapes → Bdd100k (daytime)

| Training Stage      | AP@50 | Logs & Weights |
| ------------------- | ----- | -------------- |
| `source_only`       | -     | [Source-only](#)|
| `teaching_standard` | -     | [Baseline](#)  |
| `teaching_ours`     | -     | [Ours](#)      |

**sim2city**: Sim10k → Cityscapes (car only)

| Training Stage      | AP@50 | Logs & Weights |
| ------------------- | ----- | -------------- |
| `source_only`       | -     | [Source-only](#)|
| `teaching_standard` | -     | [Baseline](#)  |
| `teaching_ours`     | -     | [Ours](#)      |

## 5. Citation

If you find our paper or code useful, please cite our work:

```bibtex
@inproceedings{borgavi2026adapting,
  title={Adapting with an Open Mind: Leveraging Open-Vocabulary Detectors for Closed Set Source-Free Domain Adaptive Object Detection},
  author={Borgavi, Kaustubh R and Shashikumar, Sarvesh and Arora, Chetan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6570--6581},
  year={2026}
}
```

## 6. Acknowledgement

This project is built upon [DRU](https://github.com/lbktrinh/DRU), [MRT-release](https://github.com/JeremyZhao1998/MRT-release), and we appreciate their excellent works and for laying a strong foundation for further research in this area.  

## 7. Contact

If you have any issues with the code or paper, feel free to contact: [aiz248319@scai.iitd.ac.in]
