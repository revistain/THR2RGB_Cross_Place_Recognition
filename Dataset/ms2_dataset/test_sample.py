import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

# MS2 Dataset Configuration
dataset_dir = '/data2/datasets/sync_data'
timestamp_list = [
    "_2021-08-06-10-59-33", "_2021-08-06-16-19-00", "_2021-08-06-17-21-04",
    "_2021-08-13-16-08-46", "_2021-08-13-16-50-57", "_2021-08-13-21-36-10",
    "_2021-08-13-22-16-02",
    "_2021-08-06-11-23-45", "_2021-08-06-16-45-28", "_2021-08-06-17-44-55",
    "_2021-08-13-16-14-48", "_2021-08-13-17-06-04", "_2021-08-13-21-58-13",
    "_2021-08-13-22-36-41",
    "_2021-08-06-11-37-46", "_2021-08-06-16-59-13", "_2021-08-13-15-46-56",
    "_2021-08-13-16-31-10", "_2021-08-13-21-18-04", "_2021-08-13-22-03-03"
]

pose_dir = '/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/Dataset/ms2_dataset/timestamp_UTM'

DB_DIS_TH = 5  # 5m for database
Query_DIS_TH = 1  # 1m for query


def get_valid_indices(pose_gt):
    """
    Get indices of valid poses (no inf or nan values).
    """
    valid_mask = ~(np.isinf(pose_gt).any(axis=1) | np.isnan(pose_gt).any(axis=1))
    return set(np.where(valid_mask)[0])


def sample_by_distance(pose_gt, distance_threshold, exclude_indices=None, valid_indices=None):
    """
    Sample poses by distance threshold using KNN.
    Returns list of sampled indices.
    """
    if exclude_indices is None:
        exclude_indices = set()

    if valid_indices is None:
        valid_indices = get_valid_indices(pose_gt)

    # Find first valid index
    start_idx = None
    for i in range(pose_gt.shape[0]):
        if i not in exclude_indices and i in valid_indices:
            start_idx = i
            break

    if start_idx is None:
        return []

    sampled_indices = [start_idx]
    sampled_poses = pose_gt[start_idx, :].reshape(1, -1)

    for i in range(start_idx + 1, pose_gt.shape[0]):
        if i in exclude_indices or i not in valid_indices:
            continue

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(sampled_poses[:, 0:2])
        dis, _ = knn.kneighbors(pose_gt[i, 0:2].reshape(1, -1), 1, return_distance=True)

        if dis > distance_threshold:
            sampled_poses = np.concatenate((sampled_poses, pose_gt[i, :].reshape(1, -1)), axis=0)
            sampled_indices.append(i)

    return sampled_indices


def count_samples_train_split(pose_pd):
    """
    STheReO_train_split 방식 (Train 중심):
    1. 5m 간격으로 DB 생성
    2. DB 제외한 나머지에서 1m 간격으로 train query 샘플링
    3. 남은 것 중 50%를 val, 50%를 test로 분배

    Returns: (db_count, train_count, val_count, test_count, total_poses, valid_poses)
    """
    pose_pd = pose_pd.iloc[1:-1, :].reset_index(drop=True)
    pose_gt = pose_pd.iloc[:, 1:3].to_numpy()
    total_poses = pose_gt.shape[0]

    # Get valid indices (no inf/nan)
    valid_indices = get_valid_indices(pose_gt)
    valid_count = len(valid_indices)

    # Create DB (5m sampling)
    db_indices = sample_by_distance(pose_gt, DB_DIS_TH, valid_indices=valid_indices)
    db_set = set(db_indices)

    # Create train query from remaining (1m sampling)
    train_indices = sample_by_distance(pose_gt, Query_DIS_TH, exclude_indices=db_set, valid_indices=valid_indices)
    train_set = set(train_indices)

    # Remaining indices for val/test (only from valid indices)
    remaining = list(valid_indices - db_set - train_set)

    if len(remaining) > 0:
        val_num = int(len(remaining) * 0.5)
        val_indices = np.random.choice(remaining, val_num, replace=False)
        test_indices = list(set(remaining) - set(val_indices))
    else:
        val_indices = []
        test_indices = []

    return len(db_indices), len(train_indices), len(val_indices), len(test_indices), total_poses, valid_count


def count_samples_test_split(pose_pd):
    """
    STheReO_test_split 방식 (Test 중심):
    1. 5m 간격으로 DB 생성
    2. DB 제외한 나머지에서 1m 간격으로 test query 샘플링
    3. 남은 것 중 50%를 val, 50%를 train으로 분배

    Returns: (db_count, train_count, val_count, test_count, total_poses, valid_poses)
    """
    pose_pd = pose_pd.iloc[1:-1, :].reset_index(drop=True)
    pose_gt = pose_pd.iloc[:, 1:3].to_numpy()
    total_poses = pose_gt.shape[0]

    # Get valid indices (no inf/nan)
    valid_indices = get_valid_indices(pose_gt)
    valid_count = len(valid_indices)

    # Create DB (5m sampling)
    db_indices = sample_by_distance(pose_gt, DB_DIS_TH, valid_indices=valid_indices)
    db_set = set(db_indices)

    # Create test query from remaining (1m sampling)
    test_indices = sample_by_distance(pose_gt, Query_DIS_TH, exclude_indices=db_set, valid_indices=valid_indices)
    test_set = set(test_indices)

    # Remaining indices for val/train (only from valid indices)
    remaining = list(valid_indices - db_set - test_set)

    if len(remaining) > 0:
        val_num = int(len(remaining) * 0.5)
        val_indices = np.random.choice(remaining, val_num, replace=False)
        train_indices = list(set(remaining) - set(val_indices))
    else:
        val_indices = []
        train_indices = []

    return len(db_indices), len(train_indices), len(val_indices), len(test_indices), total_poses, valid_count


def analyze_ms2_dataset():
    """
    Analyze MS2 dataset using both train_split and test_split methods.
    """
    print("=" * 100)
    print("MS2 Dataset Sample Count Analysis")
    print("DB: 5m sampling, Query: 1m sampling")
    print("=" * 100)

    # Train split results
    print("\n" + "=" * 120)
    print("Method 1: STheReO_train_split 방식 (1m 샘플링 -> Train, 나머지 -> Val/Test)")
    print("=" * 120)
    print(f"{'Sequence':<30} {'Total':>10} {'Valid':>10} {'DB (5m)':>10} {'Train (1m)':>12} {'Val':>10} {'Test':>10}")
    print("-" * 120)

    total_stats_train = {'total': 0, 'valid': 0, 'db': 0, 'train': 0, 'val': 0, 'test': 0}

    for timestamp in timestamp_list:
        pose_file = os.path.join(pose_dir, f'{timestamp}_timestamp_UTM.csv')
        if not os.path.exists(pose_file):
            print(f"{timestamp:<30} FILE NOT FOUND")
            continue

        pose_pd = pd.read_csv(pose_file, header=None)
        db, train, val, test, total, valid = count_samples_train_split(pose_pd)

        print(f"{timestamp:<30} {total:>10} {valid:>10} {db:>10} {train:>12} {val:>10} {test:>10}")

        total_stats_train['total'] += total
        total_stats_train['valid'] += valid
        total_stats_train['db'] += db
        total_stats_train['train'] += train
        total_stats_train['val'] += val
        total_stats_train['test'] += test

    print("-" * 120)
    print(f"{'TOTAL':<30} {total_stats_train['total']:>10} {total_stats_train['valid']:>10} {total_stats_train['db']:>10} "
          f"{total_stats_train['train']:>12} {total_stats_train['val']:>10} {total_stats_train['test']:>10}")

    # Test split results
    print("\n" + "=" * 120)
    print("Method 2: STheReO_test_split 방식 (1m 샘플링 -> Test, 나머지 -> Val/Train)")
    print("=" * 120)
    print(f"{'Sequence':<30} {'Total':>10} {'Valid':>10} {'DB (5m)':>10} {'Train':>12} {'Val':>10} {'Test (1m)':>10}")
    print("-" * 120)

    total_stats_test = {'total': 0, 'valid': 0, 'db': 0, 'train': 0, 'val': 0, 'test': 0}

    # Reset random seed for consistent results
    np.random.seed(42)

    for timestamp in timestamp_list:
        pose_file = os.path.join(pose_dir, f'{timestamp}_timestamp_UTM.csv')
        if not os.path.exists(pose_file):
            print(f"{timestamp:<30} FILE NOT FOUND")
            continue

        pose_pd = pd.read_csv(pose_file, header=None)
        db, train, val, test, total, valid = count_samples_test_split(pose_pd)

        print(f"{timestamp:<30} {total:>10} {valid:>10} {db:>10} {train:>12} {val:>10} {test:>10}")

        total_stats_test['total'] += total
        total_stats_test['valid'] += valid
        total_stats_test['db'] += db
        total_stats_test['train'] += train
        total_stats_test['val'] += val
        total_stats_test['test'] += test

    print("-" * 120)
    print(f"{'TOTAL':<30} {total_stats_test['total']:>10} {total_stats_test['valid']:>10} {total_stats_test['db']:>10} "
          f"{total_stats_test['train']:>12} {total_stats_test['val']:>10} {total_stats_test['test']:>10}")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print("\nMethod 1 (Train Split) - Train이 1m 샘플링:")
    print(f"  Total poses:        {total_stats_train['total']}")
    print(f"  Valid poses:        {total_stats_train['valid']}")
    print(f"  Total DB images:    {total_stats_train['db']}")
    print(f"  Total Train images: {total_stats_train['train']}")
    print(f"  Total Val images:   {total_stats_train['val']}")
    print(f"  Total Test images:  {total_stats_train['test']}")

    print("\nMethod 2 (Test Split) - Test가 1m 샘플링:")
    print(f"  Total poses:        {total_stats_test['total']}")
    print(f"  Valid poses:        {total_stats_test['valid']}")
    print(f"  Total DB images:    {total_stats_test['db']}")
    print(f"  Total Train images: {total_stats_test['train']}")
    print(f"  Total Val images:   {total_stats_test['val']}")
    print(f"  Total Test images:  {total_stats_test['test']}")


if __name__ == "__main__":
    analyze_ms2_dataset()



'''
load dataset: KAIST
2026-02-11 12:27:11   [Train - KAIST] Database: 1152, Queries: 12548, Total: 13700
load dataset: SNU
2026-02-11 12:27:11   [Test - SNU] Database: 1197, Queries: 12589, Total: 13786
load dataset: Valley
2026-02-11 12:27:12   [Test - Valley] Database: 179, Queries: 2082, Total: 2261

stereo는 query가 3배 많은게 정상

====================================================================================================
MS2 Dataset Sample Count Analysis
DB: 5m sampling, Query: 1m sampling
====================================================================================================

========================================================================================================================
STheReO_train_split 방식 (1m 샘플링 -> Train, 나머지 -> Val/Test)
========================================================================================================================
Sequence                            Total      Valid    DB (5m)   Train (1m)        Val       Test
------------------------------------------------------------------------------------------------------------------------
_2021-08-06-10-59-33                10440      10440       1207         4723       2255       2255
_2021-08-06-16-19-00                11866      11866       1060         3183       3811       3812
_2021-08-06-17-21-04                11455      11455       1251         3187       3508       3509
_2021-08-13-16-08-46                 2541       2541        333         1016        596        596
_2021-08-13-16-50-57                 8527       8527       1190         3137       2100       2100
_2021-08-13-21-36-10                11938      11938       1264         3797       3438       3439
_2021-08-13-22-16-02                 5232       5232        801         2104       1163       1164
_2021-08-06-11-23-45                 5808       5808        496         2080       1616       1616
_2021-08-06-16-45-28                 6670       6670        538         2215       1958       1959
_2021-08-06-17-44-55                10761      10761       1179         4467       2557       2558
_2021-08-13-16-14-48                 9165       9165        469         2211       3242       3243
_2021-08-13-17-06-04                 9687       9687       1200         4215       2136       2136
_2021-08-13-21-58-13                 2537       2537        355         1099        541        542
_2021-08-13-22-36-41                 4210       4210        784         2113        656        657
_2021-08-06-11-37-46                 9513       9513        955         2893       2832       2833
_2021-08-06-16-59-13                 6481       6481        755         1905       1910       1911
_2021-08-13-15-46-56                11810      11810       1040         3397       3686       3687
_2021-08-13-16-31-10                 5693       5693        814         1924       1477       1478
_2021-08-13-21-18-04                10129      10129       1194         4526       2204       2205
_2021-08-13-22-03-03                 7549       7549        505         2318       2363       2363
------------------------------------------------------------------------------------------------------------------------
TOTAL                              162012     162012      17390        56510      44049      44063

'''