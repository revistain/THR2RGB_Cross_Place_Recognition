import cv2
import os
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
ROOT_DIR = "/data2/datasets/sync_data"
OUTPUT_DIR = "ms2_videos_fusion"        # 저장 폴더 이름 변경
FPS = 30
RESIZE_FACTOR = 0.5                     # 50% 축소

sequence_metadata = {
    "Campus": [
        {"_2021-08-06-10-59-33": ["Clearsky", "Morning"]},
        {"_2021-08-06-17-44-55": ["Cloudy_Afterrain", "Daytime"]},
        {"_2021-08-13-17-06-04": ["Clearsky", "Daytime"]},
        {"_2021-08-13-21-18-04": ["Clearsky", "Nighttime"]}
    ],
    "Residential": [
        {"_2021-08-06-11-23-45": ["Clearsky", "Morning"]},
        {"_2021-08-06-16-45-28": ["Cloudy_Afterrain", "Daytime"]},
        {"_2021-08-13-16-14-48": ["Clearsky", "Daytime"]},
        {"_2021-08-13-22-03-03": ["Clearsky", "Nighttime"]}
    ],
    "Urban": [
        {"_2021-08-06-11-37-46": ["Clearsky", "Morning"]},
        {"_2021-08-06-16-19-00": ["Rainy", "Daytime"]},
        {"_2021-08-13-15-46-56": ["Clearsky", "Daytime"]},
        {"_2021-08-13-21-36-10": ["Clearsky", "Nighttime"]}
    ],
    "Suburban": [
        {"_2021-08-06-12-06-20": ["Clearsky", "Morning"]},
        {"_2021-08-06-17-10-27": ["Rainy", "Daytime"]},
        {"_2021-08-13-16-41-00": ["Clearsky", "Daytime"]},
        {"_2021-08-13-22-27-31": ["Clearsky", "Nighttime"]}
    ],
    "Road1": [
        {"_2021-08-06-16-59-13": ["Rainy", "Daytime"]},
        {"_2021-08-13-16-31-10": ["Clearsky", "Daytime"]},
        {"_2021-08-13-22-16-02": ["Clearsky", "Nighttime"]}
    ],
    "Road2": [
        {"_2021-08-06-17-21-04": ["Rainy", "Daytime"]},
        {"_2021-08-13-16-50-57": ["Clearsky", "Daytime"]}
    ],
    "Road3": [
        {"_2021-08-13-16-08-46": ["Clearsky", "Daytime"]},
        {"_2021-08-13-21-58-13": ["Clearsky", "Nighttime"]}
    ],
    "Road4": [
        {"_2021-08-13-22-36-41": ["Clearsky", "Nighttime"]}
    ]
}

def create_fusion_videos():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Created output directory: {os.path.abspath(OUTPUT_DIR)}")

    for location, seq_list in sequence_metadata.items():
        for seq_info in seq_list:
            timestamp_key = list(seq_info.keys())[0]
            weather, time_of_day = list(seq_info.values())[0]

            # 1. 경로 설정 (RGB & Thermal)
            rgb_folder = os.path.join(ROOT_DIR, timestamp_key, 'rgb', 'img_left')
            thr_folder = os.path.join(ROOT_DIR, timestamp_key, 'thr', 'img_left')
            
            # 경로 존재 확인
            if not os.path.exists(rgb_folder) or not os.path.exists(thr_folder):
                print(f"⚠️  Skipping {timestamp_key}: Folders not found.")
                continue

            # 2. 파일 리스트 로드 및 정렬
            rgb_images = sorted([img for img in os.listdir(rgb_folder) if img.endswith(".png")])
            thr_images = sorted([img for img in os.listdir(thr_folder) if img.endswith(".png")])

            # 개수 검증 (다르면 뒤쪽 잘림 방지위해 min 사용)
            min_len = min(len(rgb_images), len(thr_images))
            if min_len == 0:
                print(f"⚠️  Skipping {timestamp_key}: No images found.")
                continue
            
            rgb_images = rgb_images[:min_len]
            thr_images = thr_images[:min_len]

            # 3. 첫 프레임으로 사이즈 계산
            img_rgb = cv2.imread(os.path.join(rgb_folder, rgb_images[0]))
            h, w, _ = img_rgb.shape

            # 리사이즈 목표 크기 (너비, 높이는 RGB 기준 반절)
            target_w = int(w * RESIZE_FACTOR)
            target_h = int(h * RESIZE_FACTOR)

            # 4. VideoWriter 설정
            # 영상 높이는 (RGB 반절) + (Thermal 반절) = 원래 높이(h)와 같아짐 (만약 0.5배면)
            # 혹은 각각 줄여서 붙이므로 target_h * 2 가 최종 높이
            final_h = target_h * 2
            
            video_filename = f"{location}-{weather}-{time_of_day}-{timestamp_key}_fusion.mp4"
            save_path = os.path.join(OUTPUT_DIR, video_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(save_path, fourcc, FPS, (target_w, final_h))

            print(f"🎬 Processing: {video_filename}")
            print(f"   Input RGB: {rgb_folder}")
            print(f"   Output Size: {target_w}x{final_h} (Stacked)")

            # 5. 프레임 합성 루프
            for i in tqdm(range(min_len), desc="Fusion Encoding", leave=False):
                # 파일 읽기
                path_rgb = os.path.join(rgb_folder, rgb_images[i])
                path_thr = os.path.join(thr_folder, thr_images[i])

                frame_rgb = cv2.imread(path_rgb)
                # Thermal은 16bit일 수 있으므로 UNCHANGED로 읽고 정규화
                frame_thr = cv2.imread(path_thr, cv2.IMREAD_UNCHANGED)

                if frame_rgb is None or frame_thr is None:
                    continue

                # Thermal 전처리: Normalize (0~255) & Grayscale -> BGR
                frame_thr = cv2.normalize(frame_thr, None, 0, 255, cv2.NORM_MINMAX)
                frame_thr = frame_thr.astype(np.uint8)
                frame_thr = cv2.cvtColor(frame_thr, cv2.COLOR_GRAY2BGR)

                # 리사이즈 (RGB와 Thermal 모두 동일한 크기로)
                frame_rgb_small = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
                # Thermal도 RGB 사이즈에 맞춰서 리사이즈 (그래야 붙일 수 있음)
                frame_thr_small = cv2.resize(frame_thr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                # 위아래로 붙이기 (Vertical Stack)
                stacked_frame = cv2.vconcat([frame_rgb_small, frame_thr_small])

                video.write(stacked_frame)

            video.release()
            print(f"✅ Saved: {save_path}\n")

    cv2.destroyAllWindows()
    print("✨ All fusion videos generated successfully!")

if __name__ == "__main__":
    create_fusion_videos()