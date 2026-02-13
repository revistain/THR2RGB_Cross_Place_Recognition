import torch
import logging
import numpy as np
import os
import multiprocessing
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Custom Modules
from Parser import Parser
import commons
import datasets_T2R
import inference
import network
import network_only_GeM
import backbone.dinov2.block as dinoblock

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main():
    # 1. Setup & Arguments
    parser = Parser()
    args = parser.parse_arguments()
    
    commons.setup_logging(args.save_dir) 
    set_seed(args.seed)

    # 체크포인트 경로 확인
    if not args.resume:
        raise ValueError("Error: --resume flag is required. Please provide the path to the checkpoint (.pth).")
    args.resume = args.resume[0]
    
    checkpoint_path = args.resume
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logging.info(f"Use {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    logging.info(f"Comment: {args.comment}")

    # 2. Model Configuration
    model_path = Path(args.foundation_model_path)
    model_name = model_path.parts[-1].lower()
    args.features_dim = 768 if 'vitb' in model_name else 384
    dinoblock.adapter_dim = args.features_dim

    # 3. Model Initialization
    # args.use_recon_loss 값에 따라 모델 아키텍처 결정
    if args.use_recon_loss:
        logging.info("Initializing Model with Reconstruction Module...")
        model = network.CrossModalVPR_Net(
            args,
            pretrained_foundation=True,
            foundation_model_path=args.foundation_model_path,
        )
    else:
        logging.info("Initializing Model (GeM Only)...")
        model = network_only_GeM.CrossModalVPR_Net(
            args,
            pretrained_foundation=True,
            foundation_model_path=args.foundation_model_path,
        )

    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    # 4. Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # 저장된 state_dict 불러오기
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint # state_dict만 저장된 경우
        
    try:
        model.load_state_dict(state_dict)
        logging.info("Successfully loaded model weights.")
    except Exception as e:
        logging.error(f"Error loading state_dict: {e}")
        logging.warning("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # 5. Dataset Preparation
    DATASET_FOLDER = "./Dataset/save_mat"
    
    # 테스트하고 싶은 시퀀스 목록 (args.sequences에서 받음, 미지정 시 기본값 사용)
    test_sequences = args.test_seq
    
    logging.info(f"Target Sequences for Inference: {test_sequences}")

    # 6. Run Inference
    total_r1 = []

    for seq in test_sequences:
        logging.info(f"================================================")
        logging.info(f"===== Evaluating Sequence: {seq} =====")

        # 해당 시퀀스만 설정하여 데이터셋 로드
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        logging.info(f"[{seq}] Database: {test_ds.database_num}, Queries: {test_ds.queries_num}, Total: {len(test_ds)}")

        # Inference 수행
        recalls, recalls_str = inference.inference(args, test_ds, model, seq_name=seq)

        logging.info(f"Recalls for {seq}: {recalls_str}")
        total_r1.append(recalls[0]) # R@1 저장

    logging.info(f"================================================")
    logging.info(f"Average R@1 over {len(test_sequences)} sequences: {np.mean(total_r1):.2f}")
    logging.info(f"Inference Completed.")
    logging.info(f"Comment: {args.comment}")

if __name__ == "__main__":
    main()