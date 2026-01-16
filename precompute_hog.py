from Parser import Parser
import os
import numpy as np
from tqdm import tqdm
import utils
import torch
import commons
import datetime
import datasets_T2R
import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def precompute_hog_features(args, dataset, split='queries', save_dir='./hog_cache'):
    """
    HOG features를 미리 계산하고 NPY로 저장
    
    Args:
        args: Parser arguments
        dataset: Dataset object
        split: 'queries' or 'database'
        save_dir: 저장 디렉토리
    
    Returns:
        save_path: 저장된 파일 경로
    """
    # 1. 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Split에 따라 설정
    if split == 'queries':
        indices = list(range(dataset.database_num, len(dataset)))
        num_samples = dataset.queries_num
        desc = f"Computing HOG for queries"
    elif split == 'database':
        indices = list(range(dataset.database_num))
        num_samples = dataset.database_num
        desc = f"Computing HOG for database"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # 3. 파일명 생성 (dataset sequence 이름 포함)
    seq_name = '_'.join(args.sequences)
    save_path = os.path.join(save_dir, f'{split}_hog_{seq_name}.npy')
    
    # 4. 이미 존재하면 skip
    if os.path.exists(save_path):
        print(f"[SKIP] HOG features already exist: {save_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return save_path
    
    # 5. DataLoader 생성
    subset_ds = Subset(dataset, indices)
    dataloader = DataLoader(
        dataset=subset_ds,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=False,  # CPU에서 처리
        shuffle=False
    )
    
    # 6. HOG features 저장할 배열 (먼저 shape 확인 필요)
    # extract_hog_simple returns [256, 36] based on utils.py
    hog_features_all = np.empty((num_samples, 256, 36), dtype=np.float32)
    
    # 7. Batch 단위로 HOG 계산
    print(f"[START] {desc} ({num_samples} samples)")
    for inputs, batch_indices, flags in tqdm(dataloader, desc=desc, ncols=100):
        # HOG 계산
        hog_batch = utils.extract_hog_batch(inputs)  # [B, 256, 36]
        
        # 저장 인덱스 계산
        if split == 'queries':
            store_indices = batch_indices.numpy() - dataset.database_num
        else:
            store_indices = batch_indices.numpy()
        
        # 저장
        hog_features_all[store_indices] = hog_batch.cpu().numpy()
    
    # 8. NPY 저장
    np.save(save_path, hog_features_all)
    file_size_mb = hog_features_all.nbytes / 1024 / 1024
    
    print(f"[DONE] Saved HOG features to: {save_path}")
    print(f"       Shape: {hog_features_all.shape}, Size: {file_size_mb:.2f} MB")
    
    return save_path

def load_hog_features(sequences, split='queries', hog_cache_dir='./hog_cache', device='cuda'):
    """
    여러 sequence의 HOG features를 로드하고 concatenate
    """
    hog_list = []
    
    for seq in sequences:
        # 각 sequence별로 파일 찾기
        hog_filename = f'{split}_hog_{seq}.npy'
        hog_path = os.path.join(hog_cache_dir, hog_filename)
        
        if not os.path.exists(hog_path):
            raise FileNotFoundError(
                f"HOG features not found: {hog_path}\n"
                f"Please run precompute_hog.py for sequence '{seq}' first!"
            )
        
        print(f"Loading HOG from: {hog_path}")
        hog_numpy = np.load(hog_path)
        hog_list.append(hog_numpy)
    
    # Concatenate along batch dimension
    hog_all = np.concatenate(hog_list, axis=0)  # [total_samples, 256, 36]
    hog_tensor = torch.from_numpy(hog_all).float().to(device)
    
    print(f"Loaded HOG features: shape={hog_tensor.shape}, device={hog_tensor.device}")
    
    return hog_tensor

def load_hog_batch(hog_tensor, indices, offset=0):
    """
    전체 HOG tensor에서 특정 batch의 HOG만 추출
    
    Args:
        hog_tensor: torch.Tensor [num_samples, 256, 36] (from load_hog_features)
        indices: torch.Tensor or np.ndarray - batch indices
        offset: int - indices에서 빼줄 offset (e.g., database_num for queries)
    
    Returns:
        hog_batch: torch.Tensor [B, 256, 36]
    
    Example:
        >>> hog_all = load_hog_features(['SNU'], split='queries')  # [800, 256, 36]
        >>> batch_indices = torch.tensor([100, 101, 102, 103])
        >>> hog_batch = load_hog_batch(hog_all, batch_indices)
        >>> print(hog_batch.shape)  # torch.Size([4, 256, 36])
    """
    # 1. Indices를 numpy로 변환
    if isinstance(indices, torch.Tensor):
        indices_np = indices.cpu().numpy()
    else:
        indices_np = np.array(indices)
    
    # 2. Offset 적용
    indices_np = indices_np - offset
    
    # 3. Indexing
    hog_batch = hog_tensor[indices_np]
    
    return hog_batch

def save_hog_features(hog_tensor, sequences, split='queries', hog_cache_dir='./hog_cache'):
    """
    HOG tensor를 NPY로 저장
    
    Args:
        hog_tensor: torch.Tensor [num_samples, 256, 36]
        sequences: list of str
        split: 'queries' or 'database'
        hog_cache_dir: 저장 디렉토리
    
    Returns:
        save_path: 저장된 파일 경로
    
    Example:
        >>> hog_features = torch.randn(1000, 256, 36)
        >>> save_path = save_hog_features(hog_features, ['KAIST'], 'queries')
    """
    # 1. 디렉토리 생성
    os.makedirs(hog_cache_dir, exist_ok=True)
    
    # 2. 파일명 생성
    seq_name = '_'.join(sequences)
    hog_filename = f'{split}_hog_{seq_name}.npy'
    save_path = os.path.join(hog_cache_dir, hog_filename)
    
    # 3. Tensor → Numpy
    if isinstance(hog_tensor, torch.Tensor):
        hog_numpy = hog_tensor.cpu().numpy()
    else:
        hog_numpy = hog_tensor
    
    # 4. 저장
    np.save(save_path, hog_numpy)
    
    # 5. 정보 출력
    file_size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"Saved HOG features to: {save_path}")
    print(f"Shape: {hog_numpy.shape}, Size: {file_size_mb:.2f}MB")
    
    return save_path


def get_hog_cache_info(hog_cache_dir='./hog_cache'):
    """
    HOG cache 디렉토리의 모든 NPY 파일 정보 출력
    
    Args:
        hog_cache_dir: HOG 캐시 디렉토리
    
    Returns:
        info_dict: dict - {filename: (shape, size_mb)}
    
    Example:
        >>> info = get_hog_cache_info()
        >>> for filename, (shape, size) in info.items():
        >>>     print(f"{filename}: {shape}, {size:.2f}MB")
    """
    if not os.path.exists(hog_cache_dir):
        print(f"HOG cache directory not found: {hog_cache_dir}")
        return {}
    
    info_dict = {}
    
    print(f"\nHOG Cache Info ({hog_cache_dir}):")
    print("="*80)
    
    for filename in sorted(os.listdir(hog_cache_dir)):
        if filename.endswith('.npy'):
            filepath = os.path.join(hog_cache_dir, filename)
            
            # Load to get shape
            hog_np = np.load(filepath)
            shape = hog_np.shape
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            
            info_dict[filename] = (shape, size_mb)
            print(f"  {filename:40s} | Shape: {str(shape):20s} | Size: {size_mb:6.2f}MB")
    
    print("="*80)
    
    return info_dict

if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_arguments()
    print(f"Use {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    DATASET_FOLDER = "./Dataset/save_mat"
    HOG_CACHE_DIR = "./hog_cache"  # HOG 저장 디렉토리

    print("="*80)
    print("HOG Feature Pre-computation")
    print("="*80)

    # ========== KAIST Dataset (Train) ==========
    print("\n[1/3] Processing KAIST dataset...")
    args.sequences = ['KAIST']
    train_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='train')
    
    # Queries HOG
    kaist_queries_path = precompute_hog_features(
        args, train_ds, split='queries', save_dir=HOG_CACHE_DIR
    )
    
    # Database HOG (optional)
    response = input("\nCompute HOG for KAIST database? (y/n): ")
    if response.lower() == 'y':
        kaist_database_path = precompute_hog_features(
            args, train_ds, split='database', save_dir=HOG_CACHE_DIR
        )

    # ========== Test Datasets ==========
    test_sequences = ['SNU', 'Valley']
    
    for idx, seq in enumerate(test_sequences):
        print(f"\n[{idx+2}/3] Processing {seq} dataset...")
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        
        # Queries HOG
        test_queries_path = precompute_hog_features(
            args, test_ds, split='queries', save_dir=HOG_CACHE_DIR
        )
        
        # Database HOG (optional)
        response = input(f"\nCompute HOG for {seq} database? (y/n): ")
        if response.lower() == 'y':
            test_database_path = precompute_hog_features(
                args, test_ds, split='database', save_dir=HOG_CACHE_DIR
            )

    print("\n" + "="*80)
    print("HOG Pre-computation Complete!")
    print("="*80)
    print(f"\nSaved files in: {HOG_CACHE_DIR}")
    print("\nGenerated files:")
    for file in sorted(os.listdir(HOG_CACHE_DIR)):
        if file.endswith('.npy'):
            filepath = os.path.join(HOG_CACHE_DIR, file)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  - {file} ({size_mb:.2f} MB)")
    
    print("\n" + "="*80)