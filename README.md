준비작업
a. DINOv2_small pretrained
b. docker image: thr2rgb_vpr
아니면 pip install -r requirements.txt
(다운) https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
b. 데이터셋 가공(sampling)

```bash
cd Dataset
python STheReO_train_split.py
python STheReO_test_split.py
```
Train 돌리기
```bash
**- GeM 방식 학습
bash train_GeM.sh**

# --cuda_device: GPU번호
# --foundation_model_path: DINO pretrained path
# --save_dir: log파일 저장위치
# --lr: cosine warmup방식
**# --train_seq: 학습하려는 scene**
**# --test_seq: train epoch후 평가하는 scene**
# --comment: log파일이나 wandb에 기록되는 이름

# sequence 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley']
```
```bash
- CroCo 방식 학습
bash train_CroCo.sh

# --cuda_device: GPU번호
# --foundation_model_path: DINO pretrained path
# --save_dir: log파일 저장위치
# --lr: cosine warmup방식
**# --train_seq: 학습하려는 scene
# --test_seq: train epoch후 평가하는 scene**
# --comment: log파일이나 wandb에 기록되는 이름

**# --use_recon_loss: CroCo 사용여부
# --croco_mask_ratio: CroCo에서 이미지 masking하는 비율
# --recon_loss_type: 어떤 loss로 reconstruction 평가할지(mask된 부분만 평가)
# --num_decoder_depth: decoder layer 개수**

# sequence 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley']
# recon_loss_type: ['mse', 'l1', 'ssim', 'mse+ssim']
```

Test 돌리기 (pth 파일 테스트하기)
```bash
- GeM 방식 평가
bash test.sh

# --cuda_device: GPU번호
# --save_dir: log파일 저장위치
# --foundation_model_path: DINO pretrained path
**# --resume: 불러올 checkpoint <= 꼭 바꿔줘야함**
# --test_seq: train epoch후 평가하는 scene
# --comment: log파일이나 wandb에 기록되는 이름
``````
