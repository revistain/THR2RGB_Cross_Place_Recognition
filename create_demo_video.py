"""
Demo Video Generator for Cross-Modal Visual Place Recognition

메인 영상: Thermal Urban Scene 시퀀스
하단 오버레이: Query 이미지 + Top-5 예측 (reranking 전/후)

Usage:
    python create_demo_video.py \
        --resume "path/to/checkpoint.pth" \
        --sequences Urban \
        --img_time morning \
        --output_path demo_video.mp4 \
        --use_reranking recon
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
import faiss
import tempfile
import shutil

# 프로젝트 모듈
from datasets_T2R import BaseSTheReODual
from network import CrossModalVPR_Net
from reranking_dataset import RerankingDataset, reranking_collate_fn, compute_batch_recon_loss
import backbone.dinov2.block as dinoblock
from pathlib import Path


def load_model(args):
    """모델 로드"""
    # Set adapter_dim based on model type
    model_path = Path(args.foundation_model_path)
    model_name = model_path.parts[-1].lower()
    args.features_dim = 768 if 'vitb' in model_name else 384
    dinoblock.adapter_dim = args.features_dim

    model = CrossModalVPR_Net(
        args,
        pretrained_foundation=True,
        foundation_model_path=args.foundation_model_path
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # DataParallel wrapper 처리
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint: {args.resume}")

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    return model


def extract_features_and_save(model, dataloader, npy_root, seq_name, args):
    """Feature 추출 및 NPY 저장"""
    os.makedirs(npy_root, exist_ok=True)

    database_features = []
    query_features = []

    database_num = dataloader.dataset.database_num

    with torch.no_grad():
        for images, indices, flags in tqdm(dataloader, desc="Extracting features"):
            images = images.cuda()
            flags_int = [0 if f == 'rgb' else 1 for f in flags]
            flags_tensor = torch.tensor(flags_int, dtype=torch.long, device='cuda')

            # Model outputs: (final_emb, patch_emb, recon_losses, masks, cls_attn_map, ...)
            outputs = model(images, flags_tensor)
            features = outputs[0]       # global descriptor
            patch_features = outputs[1]  # patch features

            # target patches 계산 (reconstruction용)
            target_patches = patchify(images)

            for i, idx in enumerate(indices.numpy()):
                if idx < database_num:
                    # Database
                    db_idx = idx
                    np.save(
                        os.path.join(npy_root, f"Db_{seq_name}_{db_idx}.npy"),
                        patch_features[i].cpu().numpy()
                    )
                    np.save(
                        os.path.join(npy_root, f"Db_{seq_name}_target_{db_idx}.npy"),
                        target_patches[i].cpu().numpy()
                    )
                    database_features.append(features[i].cpu().numpy())
                else:
                    # Query
                    q_idx = idx - database_num
                    np.save(
                        os.path.join(npy_root, f"Query_{seq_name}_{q_idx}.npy"),
                        patch_features[i].cpu().numpy()
                    )
                    np.save(
                        os.path.join(npy_root, f"Query_{seq_name}_target_{q_idx}.npy"),
                        target_patches[i].cpu().numpy()
                    )
                    query_features.append(features[i].cpu().numpy())

    return np.array(database_features), np.array(query_features)


def patchify(imgs):
    """이미지를 패치로 분할"""
    p = 14
    B, C, H, W = imgs.shape
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)  # B, h, w, p, p, C
    x = x.reshape(B, h * w, p * p * C)
    return x


def compute_predictions(database_features, query_features, top_k=100):
    """FAISS로 predictions 계산"""
    faiss_index = faiss.IndexFlatL2(database_features.shape[1])
    faiss_index.add(database_features)
    _, predictions = faiss_index.search(query_features, top_k)
    return predictions


def compute_distances(query_utms, database_utms, predictions):
    """예측된 DB와 Query 간의 실제 거리 계산"""
    distances = np.zeros_like(predictions, dtype=np.float32)
    for q_idx in range(len(predictions)):
        q_pos = query_utms[q_idx]
        for k_idx in range(predictions.shape[1]):
            db_idx = predictions[q_idx, k_idx]
            db_pos = database_utms[db_idx]
            dist = np.linalg.norm(q_pos - db_pos)
            distances[q_idx, k_idx] = dist
    return distances


def run_reranking(model, predictions, seq_name, npy_root, args, H_feat, W_feat):
    """Reranking 수행"""
    RERANKING_TOP_K = 5

    reranking_dataset = RerankingDataset(
        predictions=predictions,
        seq_name=seq_name,
        npy_root_path=npy_root,
        reranking_top_k=RERANKING_TOP_K,
        load_targets=True
    )

    reranking_dataloader = DataLoader(
        dataset=reranking_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=reranking_collate_fn,
        pin_memory=True
    )

    predictions_list = []

    with torch.no_grad():
        for batch in tqdm(reranking_dataloader, desc="Reranking"):
            B = batch['batch_size']
            K = batch['top_k']
            BK = B * K

            thermal_feat = batch['thermal_feat'].float().cuda()
            thermal_target = batch['thermal_target'].float().cuda()
            rgb_feat = batch['rgb_feat'].float().cuda()
            rgb_target = batch['rgb_target'].float().cuda()

            N, D = thermal_feat.shape[1], thermal_feat.shape[2]

            # GeM scores
            thermal_spatial = thermal_feat.permute(0, 2, 1).view(BK, D, H_feat, W_feat)
            thermal_gem = model.module.thermal_aggregation(thermal_spatial)
            thermal_gem_scores = torch.einsum('bd,bnd->bn', thermal_gem.detach(), thermal_feat.detach())
            thermal_gem_weight = F.softmax(thermal_gem_scores, dim=-1)

            rgb_spatial = rgb_feat.permute(0, 2, 1).view(BK, D, H_feat, W_feat)
            rgb_gem = model.module.rgb_aggregation(rgb_spatial)
            rgb_gem_scores = torch.einsum('bd,bnd->bn', rgb_gem.detach(), rgb_feat.detach())
            rgb_gem_weight = F.softmax(rgb_gem_scores, dim=-1)

            # Masking
            mask_generator = model.module.mask_generator
            thermal_masks = mask_generator(thermal_feat, gem_score=thermal_gem_scores)
            rgb_masks = mask_generator(rgb_feat, gem_score=rgb_gem_scores)

            thermal_masked = model.module.mask_token.expand(BK, N, -1).clone()
            thermal_masked[~thermal_masks] = thermal_feat[~thermal_masks]

            rgb_masked = model.module.mask_token.expand(BK, N, -1).clone()
            rgb_masked[~rgb_masks] = rgb_feat[~rgb_masks]

            # Positional embedding
            thermal_masked = thermal_masked + model.module.decoder_pos_embed
            rgb_masked = rgb_masked + model.module.decoder_pos_embed
            thermal_ref = thermal_feat + model.module.decoder_pos_embed
            rgb_ref = rgb_feat + model.module.decoder_pos_embed

            # Decoder forward
            thermal_dec = thermal_masked
            for blk in model.module.decoder_blocks:
                thermal_dec = blk(thermal_dec, rgb_ref)
            thermal_dec = model.module.decoder_norm(thermal_dec)

            rgb_dec = rgb_masked
            for blk in model.module.decoder_blocks:
                rgb_dec = blk(rgb_dec, thermal_ref)
            rgb_dec = model.module.decoder_norm(rgb_dec)

            # Prediction heads
            thermal_recon = model.module.prediction_thermal_head(thermal_dec)
            rgb_recon = model.module.prediction_rgb_head(rgb_dec)

            # Reshape for batch loss
            thermal_recon = thermal_recon.view(B, K, N, -1)
            rgb_recon = rgb_recon.view(B, K, N, -1)
            thermal_masks = thermal_masks.view(B, K, N)
            rgb_masks = rgb_masks.view(B, K, N)
            thermal_target = thermal_target.view(B, K, N, -1)
            rgb_target = rgb_target.view(B, K, N, -1)
            thermal_gem_weight = thermal_gem_weight.view(B, K, N)
            rgb_gem_weight = rgb_gem_weight.view(B, K, N)

            # Vectorized loss computation
            recon_losses = compute_batch_recon_loss(
                thermal_recon, rgb_recon,
                thermal_masks, rgb_masks,
                thermal_target, rgb_target,
                thermal_gem_weight, rgb_gem_weight,
                use_weight=args.use_gem_recon_weight
            )

            # Rerank
            rerank_indices = recon_losses.argsort(dim=1)

            top_k_indices = batch['top_k_indices']
            for i in range(B):
                reranked = top_k_indices[i][rerank_indices[i].cpu().numpy()]
                # 나머지 predictions는 원래 순서 유지
                full_pred = predictions[batch['query_indices'][i]].copy()
                full_pred[:RERANKING_TOP_K] = reranked
                predictions_list.append(full_pred)

    return np.array(predictions_list)


def create_result_panel(
    query_img,
    db_images,
    distances,
    is_positive,
    panel_height=120,
    title="Predictions"
):
    """결과 패널 생성"""
    K = len(db_images)
    img_width = int(panel_height * 1.5)

    # Query 이미지 리사이즈
    query_resized = cv2.resize(query_img, (img_width, panel_height))

    # DB 이미지들 리사이즈 및 테두리 추가
    db_panels = []
    for i, (db_img, dist, is_pos) in enumerate(zip(db_images, distances, is_positive)):
        db_resized = cv2.resize(db_img, (img_width, panel_height))

        # 테두리 색상 (positive: 초록, negative: 빨강)
        border_color = (0, 255, 0) if is_pos else (0, 0, 255)
        border_thickness = 4

        cv2.rectangle(
            db_resized,
            (0, 0),
            (img_width - 1, panel_height - 1),
            border_color,
            border_thickness
        )

        # 거리 텍스트
        dist_text = f"{dist:.1f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(dist_text, font, font_scale, thickness)
        cv2.rectangle(
            db_resized,
            (5, panel_height - text_h - 12),
            (12 + text_w, panel_height - 5),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            db_resized,
            dist_text,
            (7, panel_height - 10),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        # 순위 표시
        rank_text = f"#{i+1}"
        cv2.putText(
            db_resized,
            rank_text,
            (7, 20),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        db_panels.append(db_resized)

    # Query 라벨
    query_with_label = query_resized.copy()
    cv2.putText(
        query_with_label,
        "Query",
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # 화살표 영역
    arrow_width = 30
    arrow_panel = np.zeros((panel_height, arrow_width, 3), dtype=np.uint8)
    arrow_center = (arrow_width // 2, panel_height // 2)
    cv2.arrowedLine(
        arrow_panel,
        (5, arrow_center[1]),
        (arrow_width - 5, arrow_center[1]),
        (255, 255, 255),
        2,
        tipLength=0.4
    )

    # 전체 패널 조합
    panel = np.hstack([query_with_label, arrow_panel] + db_panels)

    # 제목 추가
    title_height = 25
    title_panel = np.zeros((title_height, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title_panel,
        title,
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    return np.vstack([title_panel, panel])


class DemoVideoGenerator:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.dataset = None
        self.database_features = None
        self.query_features = None
        self.predictions_before = None
        self.predictions_after = None
        self.distances_before = None
        self.distances_after = None
        self.npy_root = None
        self.seq_name = args.sequences[0]

    def setup(self):
        """모델과 데이터셋 설정"""
        print("Loading model...")
        self.model = load_model(self.args)

        print("Loading dataset...")
        self.dataset = BaseSTheReODual(self.args, self.args.datasets_folder, split='test')

        print(f"Dataset type: {self.dataset.dataset_type}")
        print(f"Database: {self.dataset.database_num} images")
        print(f"Queries: {self.dataset.queries_num} images")

        # 임시 NPY 디렉토리
        self.npy_root = tempfile.mkdtemp(prefix="demo_npy_")
        print(f"NPY temp dir: {self.npy_root}")

    def run_inference(self):
        """Inference 실행"""
        # DataLoader 설정
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.infer_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        print("Extracting features...")
        self.database_features, self.query_features = extract_features_and_save(
            self.model, dataloader, self.npy_root, self.seq_name, self.args
        )

        print("Computing predictions...")
        self.predictions_before = compute_predictions(
            self.database_features,
            self.query_features,
            top_k=max(self.args.recall_values)
        )

        # Reranking 전 거리 계산
        self.distances_before = compute_distances(
            self.dataset.queries_utms,
            self.dataset.database_utms,
            self.predictions_before
        )

        # Reranking
        if self.args.use_reranking != "none":
            print(f"Running reranking ({self.args.use_reranking})...")
            H_feat = int(self.args.resize[0] / 14)
            W_feat = int(self.args.resize[1] / 14)

            self.predictions_after = run_reranking(
                self.model,
                self.predictions_before,
                self.seq_name,
                self.npy_root,
                self.args,
                H_feat,
                W_feat
            )

            self.distances_after = compute_distances(
                self.dataset.queries_utms,
                self.dataset.database_utms,
                self.predictions_after
            )
        else:
            self.predictions_after = self.predictions_before.copy()
            self.distances_after = self.distances_before.copy()

        print("Inference complete!")

    def get_thermal_image(self, path):
        """Thermal 이미지 로드"""
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def get_rgb_image(self, path):
        """RGB 이미지 로드"""
        if self.dataset.dataset_type == 'sthereo':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:  # ms2
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img

    def create_frame(self, query_idx, main_frame_size=(1280, 720)):
        """단일 프레임 생성"""
        width, height = main_frame_size
        top_k = 5
        panel_height = 100

        # 메인 thermal 이미지
        thermal_path = self.dataset.t_queries_paths[query_idx]
        main_thermal = self.get_thermal_image(thermal_path)
        if main_thermal is None:
            main_thermal = np.zeros((height // 2, width, 3), dtype=np.uint8)

        # 메인 영역 높이
        main_height = height - (panel_height + 25) * 2 - 20
        main_thermal = cv2.resize(main_thermal, (width, main_height))

        # Query 이미지 (패널용)
        query_img = self.get_thermal_image(thermal_path)
        if query_img is None:
            query_img = np.zeros((panel_height, panel_height, 3), dtype=np.uint8)

        # Top-K database 이미지 로드
        db_images_before = []
        for db_idx in self.predictions_before[query_idx, :top_k]:
            db_path = self.dataset.rgb_database_paths[db_idx]
            db_img = self.get_rgb_image(db_path)
            if db_img is None:
                db_img = np.zeros((panel_height, panel_height, 3), dtype=np.uint8)
            db_images_before.append(db_img)

        db_images_after = []
        for db_idx in self.predictions_after[query_idx, :top_k]:
            db_path = self.dataset.rgb_database_paths[db_idx]
            db_img = self.get_rgb_image(db_path)
            if db_img is None:
                db_img = np.zeros((panel_height, panel_height, 3), dtype=np.uint8)
            db_images_after.append(db_img)

        # Positive 판정
        positive_threshold = self.args.soft_positives_dist_threshold
        is_positive_before = self.distances_before[query_idx, :top_k] < positive_threshold
        is_positive_after = self.distances_after[query_idx, :top_k] < positive_threshold

        # 패널 생성
        panel_before = create_result_panel(
            query_img,
            db_images_before,
            self.distances_before[query_idx, :top_k],
            is_positive_before,
            panel_height=panel_height,
            title="Before Reranking (Global Descriptor)"
        )

        panel_after = create_result_panel(
            query_img,
            db_images_after,
            self.distances_after[query_idx, :top_k],
            is_positive_after,
            panel_height=panel_height,
            title="After Reranking (Reconstruction-based)"
        )

        # 패널 너비 조정
        panel_before = cv2.resize(panel_before, (width, panel_before.shape[0]))
        panel_after = cv2.resize(panel_after, (width, panel_after.shape[0]))

        # Query 정보 오버레이
        info_text = f"Query #{query_idx + 1} / {self.dataset.queries_num}"
        cv2.putText(
            main_thermal,
            info_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Top-1 결과 표시
        top1_before = self.distances_before[query_idx, 0]
        top1_after = self.distances_after[query_idx, 0]
        result_text = f"Top-1: {top1_before:.1f}m -> {top1_after:.1f}m"
        color = (0, 255, 0) if top1_after < positive_threshold else (0, 0, 255)
        cv2.putText(
            main_thermal,
            result_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # 전체 프레임 조합
        separator = np.zeros((5, width, 3), dtype=np.uint8)
        frame = np.vstack([
            main_thermal,
            separator,
            panel_before,
            separator,
            panel_after
        ])

        # 최종 크기 조정
        frame = cv2.resize(frame, main_frame_size)

        return frame

    def generate_video(self, output_path, fps=10):
        """전체 영상 생성"""
        frame_size = (1280, 720)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        print(f"Generating video: {output_path}")
        print(f"Total frames: {self.dataset.queries_num}")

        for query_idx in tqdm(range(self.dataset.queries_num), desc="Creating frames"):
            frame = self.create_frame(query_idx, frame_size)
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved: {output_path}")

    def cleanup(self):
        """임시 파일 정리"""
        if self.npy_root and os.path.exists(self.npy_root):
            shutil.rmtree(self.npy_root)
            print(f"Cleaned up temp dir: {self.npy_root}")


def parse_args():
    """인자 파싱"""
    parser = argparse.ArgumentParser(description="Demo Video Generator")

    # 필수 인자
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--sequences", type=str, nargs="+", default=["Urban"],
                        help="Dataset sequences to use")
    parser.add_argument("--img_time", type=str, default="morning",
                        choices=["morning", "evening", "allday", "nighttime", "clearsky", "rainy", "daytime", "latetime"],
                        help="Time of day for queries")

    # 출력 설정
    parser.add_argument("--output_path", type=str, default="demo_video.mp4",
                        help="Output video path")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video frame rate")

    # 모델/데이터셋 설정
    parser.add_argument("--datasets_folder", type=str,
                        default="/home/jwkim/workspace/Dataset/save_mat",
                        help="Dataset folder path")
    parser.add_argument("--resize", type=int, nargs=2, default=[224, 224],
                        help="Image resize dimensions")
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="Recall values")
    parser.add_argument("--test_method", type=str, default="hard_resize",
                        help="Test method")
    parser.add_argument("--soft_positives_dist_threshold", type=float, default=25.0,
                        help="Positive threshold in meters")

    # 모델 설정
    parser.add_argument("--foundation_model_path", type=str,
                        default="backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth",
                        help="Foundation model path")
    parser.add_argument("--features_dim", type=int, default=768)
    parser.add_argument("--use_dino_decoder", action="store_true", default=True)
    parser.add_argument("--dino_decoder_layer_start", type=int, default=6)
    parser.add_argument("--dino_decoder_layer_end", type=int, default=12)
    parser.add_argument("--unfreeze_dino_decoder", action="store_true", default=False)
    parser.add_argument("--is_dino_dec_stage2", action="store_true", default=False)
    parser.add_argument("--num_decoder_depth", type=int, default=8)
    parser.add_argument("--croco_mask_ratio", type=float, default=0.8)
    parser.add_argument("--use_reranking", type=str, default="recon",
                        choices=["none", "recon", "distance"])
    parser.add_argument("--use_recon_loss", action="store_true", default=True)
    parser.add_argument("--recon_loss_type", type=str, default="mse")
    parser.add_argument("--num_trainable_blocks_RGB", type=int, default=2)
    parser.add_argument("--num_trainable_blocks_THERMAL", type=int, default=2)
    parser.add_argument("--use_only_cross_decoder", action="store_true", default=False)
    parser.add_argument("--use_sela_local_loss", action="store_true", default=False)
    parser.add_argument("--use_diff_loss", action="store_true", default=False)
    parser.add_argument("--use_mlp_dim_before_decoder", type=int, default=0)
    parser.add_argument("--masking_method", type=str, default="random")
    parser.add_argument("--use_gem_recon_weight", action="store_true", default=False)
    parser.add_argument("--use_swin_decoder", action="store_true", default=False)
    parser.add_argument("--swin_window_size", type=int, default=4)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_distance_loss", action="store_true", default=False)
    parser.add_argument("--distance_tau", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")

    # 추가 모델 설정
    parser.add_argument("--use_cls_for_vpr", action="store_true", default=False)
    parser.add_argument("--r2_penultimate_layer", action="store_true", default=False)
    parser.add_argument("--selaVPR_rerank_score_type", type=str, default="none")
    parser.add_argument("--visualize_attention", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    generator = DemoVideoGenerator(args)
    try:
        generator.setup()
        generator.run_inference()
        generator.generate_video(args.output_path, args.fps)
    finally:
        generator.cleanup()

    print("Done!")


if __name__ == "__main__":
    main()
