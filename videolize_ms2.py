import cv2
import os
from tqdm import tqdm
import numpy as np

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
ROOT_DIR = "/data2/datasets/sync_data"  # 실제 데이터셋 루트 경로
OUTPUT_DIR = "ms2_videos"               # 저장할 폴더 이름
FPS = 30                                # 초당 프레임 수
RESIZE_FACTOR = 0.5                     # 0.5 = 50% 크기로 축소

# 제공해주신 메타데이터
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

def create_videos():
    # 저장 경로 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Created output directory: {os.path.abspath(OUTPUT_DIR)}")

    # Metadata 순회 (Location -> List -> Dict)
    for location, seq_list in sequence_metadata.items():
        for seq_info in seq_list:
            # seq_info 예시: {"_2021-08-06...": ["Clearsky", "Morning"]}
            timestamp_key = list(seq_info.keys())[0]
            weather, time_of_day = list(seq_info.values())[0]

            # 1. 입력 이미지 경로 구성 (root / timestamp / rgb / img_left)
            input_folder = os.path.join(ROOT_DIR, timestamp_key, 'rgb', 'img_left')
            
            if not os.path.exists(input_folder):
                print(f"⚠️  Skipping {timestamp_key}: Path not found ({input_folder})")
                continue

            # 2. 이미지 리스트 로드 및 정렬
            images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
            images.sort() # 000001.png 꼴이므로 기본 sort 사용

            if not images:
                print(f"⚠️  Skipping {timestamp_key}: No images found.")
                continue

            # 3. 첫 프레임 읽어서 원본 크기 확인
            first_frame_path = os.path.join(input_folder, images[0])
            frame = cv2.imread(first_frame_path)
            h, w, _ = frame.shape

            # 4. 리사이즈 크기 계산 (정수형 변환 필수)
            new_w = int(w * RESIZE_FACTOR)
            new_h = int(h * RESIZE_FACTOR)

            # 5. 비디오 파일명 생성 (요청하신 포맷)
            # 포맷: Location-Weather-Time-Timestamp.mp4
            video_filename = f"{location}-{weather}-{time_of_day}-{timestamp_key}.mp4"
            save_path = os.path.join(OUTPUT_DIR, video_filename)

            # 6. VideoWriter 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 주의: VideoWriter는 (width, height) 순서로 받음
            video = cv2.VideoWriter(save_path, fourcc, FPS, (new_w, new_h))

            print(f"🎬 Processing: {video_filename}")
            print(f"   Input: {input_folder}")
            print(f"   Resize: ({w}x{h}) -> ({new_w}x{new_h})")

            # 7. 프레임 변환 및 저장
            for img_name in tqdm(images, desc="Encoding", leave=False):
                img_path = os.path.join(input_folder, img_name)
                frame = cv2.imread(img_path)
                
                if frame is None:
                    continue

                # 핵심: 리사이즈 수행
                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                video.write(resized_frame)

            video.release()
            print(f"✅ Saved to: {save_path}\n")

    cv2.destroyAllWindows()
    print("✨ All videos generated successfully!")

if __name__ == "__main__":
    create_videos()