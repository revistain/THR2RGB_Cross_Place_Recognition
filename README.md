# 준비 작업

## a. DINOv2_small pretrained 다운로드
- **Docker image**: `thr2rgb_vpr`
- 또는 `pip install -r requirements.txt`
- **모델 다운로드**: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

## b. 데이터셋 가공 (sampling)
```bash
cd Dataset
python sthereo_dataset/STheReO_train_split.py
python sthereo_dataset/STheReO_test_split.py
python ms2_dataset/ms2_train_test.py
```

---

# Train 돌리기

## Wandb 설정
- train_wandb.py에 wandb.init 설정하기

## GeM 방식 학습
```bash
bash train_GeM.sh
```

**주요 파라미터**:
- `--cuda_device`: GPU 번호
- `--foundation_model_path`: DINO pretrained path
- `--save_dir`: log 파일 저장 위치
- `--lr`: cosine warmup 방식
- **`--train_seq`**: 학습하려는 scene
- **`--test_seq`**: train epoch 후 평가하는 scene
- `--comment`: log 파일이나 wandb에 기록되는 이름

**sequence 종류**: `['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley']`

## CroCo 방식 학습
```bash
bash train_CroCo.sh
```

**주요 파라미터**:
- `--cuda_device`: GPU 번호
- `--foundation_model_path`: DINO pretrained path
- `--save_dir`: log 파일 저장 위치
- `--lr`: cosine warmup 방식
- **`--train_seq`**: 학습하려는 scene
- **`--test_seq`**: train epoch 후 평가하는 scene
- `--comment`: log 파일이나 wandb에 기록되는 이름
- **`--use_recon_loss`**: CroCo 사용 여부
- **`--croco_mask_ratio`**: CroCo에서 이미지 masking하는 비율
- **`--recon_loss_type`**: 어떤 loss로 reconstruction 평가할지 (mask된 부분만 평가)
- **`--num_decoder_depth`**: decoder layer 개수

**sequence 종류**: `['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley']`  
**recon_loss_type**: `['mse', 'l1', 'ssim', 'mse+ssim']`

---

# Test 돌리기 (pth 파일 테스트하기)

## GeM 방식 평가
```bash
bash test.sh
```

**주요 파라미터**:
- `--cuda_device`: GPU 번호
- `--save_dir`: log 파일 저장 위치
- `--foundation_model_path`: DINO pretrained path
- **`--resume`**: 불러올 checkpoint **← 꼭 바꿔줘야 함**
- `--test_seq`: train epoch 후 평가하는 scene
- `--comment`: log 파일이나 wandb에 기록되는 이름

---
**파일들**
- train_wandb.py : main 함수
- [network.py](http://network.py): CroCo 방식 model 정의
- network_only_GeM.py: GeM 방식 model 정의
- datasets_T2R.py: dataloader 및 dataset
- inference.py: 추론
- fast_inference.py: 이미 학습 끝난 pth를 불러서 inference만 할때 사용
