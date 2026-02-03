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
import utils
import datasets_T2R
import inference
import network
import network_only_GeM
import backbone.dinov2.block as dinoblock
from visual import visualize_cls_gem_attention_from_outputs

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
    
    # 로깅 설정 (콘솔 출력만 하거나, 별도 로그 파일 저장)
    # 로깅 설정 (콘솔 출력만 하거나, 별도 로그 파일 저장)
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

    # 2. Model Configuration (Train 코드와 동일하게 설정)
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
    # Train 시 DataParallel로 저장되었으므로 여기서도 감싸줍니다.
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
    
    # 테스트하고 싶은 시퀀스 목록 (기본값: SNU, Valley)
    # 필요시 args.sequences를 덮어쓰거나 파라미터로 조절 가능
    target_sequences = ['Valley', 'SNU'] 
    
    logging.info(f"Target Sequences for Inference: {target_sequences}")

    # 6. Run Inference
    total_r1 = []

    for seq in target_sequences:
        logging.info(f"================================================")
        logging.info(f"===== Evaluating Sequence: {seq} =====")

        # 해당 시퀀스만 설정하여 데이터셋 로드
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        logging.info(f"[{seq}] Database: {test_ds.database_num}, Queries: {test_ds.queries_num}, Total: {len(test_ds)}")

        # Visualize attention maps if requested
        if args.visualize_attention:
            from torch.utils.data import DataLoader
            from torch.utils.data.dataset import Subset
            import random

            vis_save_dir = f"./attention_visualizations/{args.comment}_{seq}"
            os.makedirs(vis_save_dir, exist_ok=True)

            # Sample some images for visualization
            num_vis_samples = min(4, len(test_ds))
            vis_indices = random.sample(range(len(test_ds)), num_vis_samples)

            with torch.no_grad():
                # Visualize thermal queries
                thermal_indices = [i for i in vis_indices if i >= test_ds.database_num]
                if thermal_indices:
                    thermal_subset = Subset(test_ds, thermal_indices)
                    thermal_loader = DataLoader(thermal_subset, batch_size=len(thermal_indices), shuffle=False)

                    for inputs, indices, flags in thermal_loader:
                        flags_int = [1 if f == 'rgb' else 0 for f in flags]
                        flags_tensor = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                        outputs = model(inputs.to(args.device), flags_tensor)

                        # Get multi-head attention from backbone for PCA
                        backbone_out = model.module.shared_backbone(inputs.to(args.device), return_attention=True)
                        cls_attn_multihead = backbone_out.get("cls_attention", None)  # [B, num_heads, N]

                        global_desc = outputs[0]   # [B, D]
                        patch_tokens = outputs[1]  # [B, N, D]
                        cls_attn = outputs[4]      # [B, N]
                        gem_attn = outputs[7]      # [B, N]

                        save_path = os.path.join(vis_save_dir, f"thermal_queries_cls_gem_attention.png")
                        visualize_cls_gem_attention_from_outputs(
                            inputs, cls_attn, gem_attn, args.device, save_path, modality='thermal',
                            patch_tokens=patch_tokens,
                            cls_attn_multihead=cls_attn_multihead,
                            global_desc=global_desc
                        )
                        break

                # Visualize RGB database
                rgb_indices = [i for i in vis_indices if i < test_ds.database_num]
                if not rgb_indices:
                    rgb_indices = random.sample(range(test_ds.database_num), min(4, test_ds.database_num))

                rgb_subset = Subset(test_ds, rgb_indices)
                rgb_loader = DataLoader(rgb_subset, batch_size=len(rgb_indices), shuffle=False)

                for inputs, indices, flags in rgb_loader:
                    flags_int = [1 if f == 'rgb' else 0 for f in flags]
                    flags_tensor = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                    outputs = model(inputs.to(args.device), flags_tensor)

                    # Get multi-head attention from backbone for PCA
                    backbone_out = model.module.shared_backbone(inputs.to(args.device), return_attention=True)
                    cls_attn_multihead = backbone_out.get("cls_attention", None)  # [B, num_heads, N]

                    global_desc = outputs[0]   # [B, D]
                    patch_tokens = outputs[1]  # [B, N, D]
                    cls_attn = outputs[4]      # [B, N]
                    gem_attn = outputs[7]      # [B, N]

                    save_path = os.path.join(vis_save_dir, f"rgb_database_cls_gem_attention.png")
                    visualize_cls_gem_attention_from_outputs(
                        inputs, cls_attn, gem_attn, args.device, save_path, modality='rgb',
                        patch_tokens=patch_tokens,
                        cls_attn_multihead=cls_attn_multihead,
                        global_desc=global_desc
                    )
                    break

            logging.info(f"Saved attention visualizations to {vis_save_dir}")

        # Inference 수행
        recalls, recalls_str = inference.inference(args, test_ds, model, seq_name=seq)

        logging.info(f"Recalls for {seq}: {recalls_str}")
        total_r1.append(recalls[0]) # R@1 저장

    logging.info(f"================================================")
    logging.info(f"Average R@1 over {len(target_sequences)} sequences: {np.mean(total_r1):.2f}")
    logging.info(f"Inference Completed.")
    logging.info(f"Comment: {args.comment}")

if __name__ == "__main__":
    main()