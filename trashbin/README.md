# 정리한 결과보고서!!
- https://www.notion.so/4-rgb-t-2bcb3ee9d68780e9bb82d8376a8cf149?source=copy_link

# 혁신인재 4차년도, Cross-spectral PR
- Query는 thermal image, DB는 RGB image

## 사용법
- 아래 원본 코드와 동일

## Dataset
### 구성
- STheReO dataset 의 **3개 Sequence (KAIST, SNU, Valley)** 를 사용
- 각 Sequence 는 **daytime**(morning, afternoon)과 **nighttime**(evening) 로 구성됨.
- DB 는 morning 으로 구성되고, Query는 전체 alltime (morning+afternoon+evening) 으로 구성
### 학습 및 평가
- 학습은 KAIST, 평가는 SNU, Valley 에서 zero-shot 으로 진행

## 실험 결과
- Baseline 1: DINOv2 전체 freeze, adapter만 학습, RGBT-VPR논문과 동일 (num_trainable_blocks=0)
- Baseline 2: DINOv2 blocks 중 뒤쪽 4개 & adapter 학습 (num_trainable_blocks=4)

### SNU Sequence
| 방법론 | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| Baseline 1 | 63.5 | 84.7 | 89.8 |
| Baseline 2 | 67.8 | 87.3 | 92.3 |

### Valley Sequence
| 방법론 | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| Baseline 1 | 88.3 | 98.9 | 99.9 |
| Baseline 2 | 86.2 | 97.3 | 98.8 |

---------------------------------------------
# 아래는 원본 논문의 README
# RGBT-VPR: RGB-Thermal Visual Place Recognition via Vision Foundation Model
This is the official repository for the IROS 2025 paper: [RGB-Thermal Visual Place Recognition via Vision Foundation Model]()

## Abstract
*Visual place recognition is a critical component of robust simultaneous localization and mapping systems. Conventional approaches primarily rely on RGB imagery, but their performance degrades significantly in extreme environments, such as those with poor illumination and airborne particulate interference (e.g., smoke or fog), which significantly degrade the performance of RGB-based methods. Furthermore, existing techniques often struggle with cross-scenario generalization.
To overcome these limitations, we propose an RGB-thermal multimodal fusion framework for place recognition, specifically designed to enhance robustness in extreme environmental conditions. Our framework incorporates a dynamic RGB-thermal fusion module, coupled with dual fine-tuned vision foundation models as the feature extraction backbone. Experimental results on public datasets and our self-collected dataset demonstrate that our method significantly outperforms state-of-the-art RGB-based approaches, achieving generalizable and robust retrieval capabilities across day and night scenarios.*

## Getting started
### Try our model
You can run the `quickstart.ipynb` to try using our model for visual place recognition.

You can download the checkpoint [HERE](https://github.com/HITSZ-NRSL/RGB-Thermal-VPR/releases/tag/v1.0.0)
### Prepare Data
Our network is trained and evaluated on the [SThReO Dataset](https://sites.google.com/view/rpmsthereo/). You should first download the dataset [HERE](https://sites.google.com/view/rpmsthereo/download).

After that, run the `Dataset/STheReO_train_split.ipynb` and `Dataset/STheReO_test_split.ipynb` to split the original dataset and get the `.mat` file.

### Train the model
Our model use pretrained [DINOv2](https://dinov2.metademolab.com/) as backbone, so download pretrained weights of ViT-B/14 size from [HERE](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models) before training our model.

Then you can start training by following command:
```
python3 train.py --save_dir /path/to/your/save/directory features_dim 768 --sequences KAIST --foundation_model_path /path/to/pretrained/weights
```

### Evaluation
You can evaluate the performance by following command:
```
python3 eval.py --resume /path/to/checkpoint --save_dir /path/to/save/directory --sequences SNU --img_time allday --features_dim 768
```
* `sequences`: choose from {SNU, Valley}
* `img_time`: choose from {daytime, nighttime, allday}

# Video

[![](https://img.youtube.com/vi/DdcS2P67XFQ/hqdefault.jpg)](https://youtu.be/DdcS2P67XFQ)
