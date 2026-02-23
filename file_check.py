DATA_DIR = "/data2/datasets/sync_data"

from pathlib import Path

def save_relative_paths_to_txt(target_dir, output_file):
    base_path = Path(target_dir)
    
    # .relative_to(base_path)를 사용하여 상대 경로 추출
    # f.is_file()로 폴더는 제외하고 파일만 수집
    rel_paths = [str(f.relative_to(base_path)) for f in base_path.rglob('*') if f.is_file()]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in sorted(rel_paths):
            f.write(path + '\n')
            
    print(f"기준 폴더: {target_dir}")
    print(f"총 {len(rel_paths)}개의 상대 경로를 '{output_file}'에 저장했습니다.")

# 사용 예시
save_relative_paths_to_txt(DATA_DIR, 'file_list.txt')