import os
import scipy.io as io
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from enum import Enum

np.random.seed(42)

class SCENE(Enum):
    Morning = 0
    Clearsky = 1
    Rainy = 2
    Nighttime = 3


dataset_dir = '/DATA2/datasets/stereo_dataset/MS2/sync_data'
# 전부 있는 Scene (Campus, Residential, Urban, SubUrban(얘는 또 dataset에 없음...))
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

seqs = ["Campus", "Residential", "Urban"]
for seq in seqs:
    print(f"  ==== MS2 DATASET mat file generating :: {seq} ====")
    # seq 재정렬
    # [Morning, Daytime(Clearsky), Daytime(Rainy), Nighttime]
    seq_list = [None for _ in range(4)]
    for seq_dict in sequence_metadata[seq]:
        seq_name = str(list(seq_dict.keys())[0])
        weather, time = list(seq_dict.values())[0]
        if time == "Morning":
            seq_list[0] = seq_name
        elif time == "Daytime":
            if weather == "Clearsky":
                seq_list[1] = seq_name
            elif weather in ["Rainy", "Cloudy_Afterrain"]:
                seq_list[2] = seq_name
        elif time == "Nighttime":
            seq_list[3] = seq_name
        else:
            raise ValueError("Not a Valid Time in sequence")
    assert None not in seq_list
    
    # 기존 코드
    pose_dir = '/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/Dataset/ms2_dataset/timestamp_UTM'
    dataset_path = [os.path.join(dataset_dir, index) for index in seq_list]    
    pose_path = [pose_dir + f'/{path}_timestamp_UTM.csv' for path in seq_list]  
    rgb_path = [path + '/rgb/img_left/' for path in dataset_path]   
    thermal_path = [path + '/thr/img_left/' for path in dataset_path]   

    morning_pose_pd = pd.read_csv(pose_path[0], header=None)
    clearsky_pose_pd = pd.read_csv(pose_path[1], header=None)
    rainy_pose_pd = pd.read_csv(pose_path[2], header=None)
    nighttime_pose_pd = pd.read_csv(pose_path[3], header=None)

    DB_DIS_TH = 5
    Query_DIS_TH = 1

    save_path_train = os.path.join('save_mat', 'train', seq)
    save_path_test = os.path.join('save_mat', 'test', seq)
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)
        print(f"- Create {save_path_train} ")
        
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)
        print(f"- Create {save_path_test} ")

    ## create morning database
    db_index = [0]
    morning_pose_pd = morning_pose_pd.iloc[1:-1, :].reset_index(drop=True)
    morning_pose_gt = morning_pose_pd.iloc[:, 1:3].to_numpy()
    morning_db = morning_pose_gt[0, :].reshape(1, -1)  # add the first frame

    for i in range(1, morning_pose_gt.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(morning_db[:, 0:2])
        dis, index = knn.kneighbors(morning_pose_gt[i, 0:2].reshape(1, -1), 1, return_distance=True)

        if dis > DB_DIS_TH:
            morning_db = np.concatenate((morning_db, morning_pose_gt[i, :].reshape(1, -1)), axis=0)
            db_index.append(i)

    print(f"- Morning dataset  has {morning_pose_gt.shape[0]} poses")
    print(f"- Morning database has {morning_db.shape[0]} poses")

    morning_db_rgb = [os.path.join(rgb_path[SCENE.Morning.value], f"{idx:06}.png") for idx in db_index]
    morning_db_t   = [os.path.join(thermal_path[SCENE.Morning.value], f"{idx:06}.png") for idx in db_index]

    ## Create Morning Query
    morning_index = [i for i in range(morning_pose_gt.shape[0])]
    morning_index = list(set(morning_index) - set(db_index))

    morning_train_index = [morning_index[0]]
    morning_train_pose = morning_pose_gt[morning_train_index, :].reshape(1, -1)  # add the first frame

    for i in range(1, morning_pose_gt.shape[0]):
        if i not in db_index:
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(morning_train_pose[:, 0:2])
            dis, index = knn.kneighbors(morning_pose_gt[i, 0:2].reshape(1, -1), 1, return_distance=True)
        
            if dis > Query_DIS_TH:
                morning_train_pose = np.concatenate((morning_train_pose, morning_pose_gt[i, :].reshape(1, -1)), axis=0)
                morning_train_index.append(i)

    train_q_index = morning_train_index
    index = list(set(morning_index) - set(train_q_index))
    
    val_q_num = int(len(index) *0.5)
    val_q_index = np.random.choice(list(set(morning_index)-set(train_q_index)), val_q_num, replace=False)
    test_q_index = list(set(morning_index) - set(train_q_index) - set(val_q_index))

    morning_q_rgb_train = [os.path.join(rgb_path[SCENE.Morning.value], f"{idx:06}") + '.png' for idx in train_q_index]
    morning_q_t_train = [os.path.join(thermal_path[SCENE.Morning.value], f"{idx:06}") + '.png' for idx in train_q_index]
    morning_q_pose_train = morning_pose_gt[train_q_index,:]
    print(f"- Created Morning Query \t: {len(morning_q_rgb_train)}")

    ## create clearsky query
    clearsky_pose_pd = clearsky_pose_pd.iloc[1:-1, :].reset_index(drop=True)
    clearsky_pose_gt = clearsky_pose_pd.iloc[:, 1:3].to_numpy()
    clearsky_index = [i for i in range(clearsky_pose_gt.shape[0])]

    clearsky_train_index = [0]
    clearsky_train_pose = clearsky_pose_gt[0, :].reshape(1, -1)  # add the first frame

    for i in range(1, clearsky_pose_gt.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(clearsky_train_pose[:, 0:2])
        dis, index = knn.kneighbors(clearsky_pose_gt[i, 0:2].reshape(1, -1), 1, return_distance=True)

        if dis > Query_DIS_TH:
            clearsky_train_pose = np.concatenate((clearsky_train_pose, clearsky_pose_gt[i, :].reshape(1, -1)), axis=0)
            clearsky_train_index.append(i)

    train_q_index = clearsky_train_index
    index = list(set(clearsky_index)- set(train_q_index))
    
    val_q_num = int(len(index) *0.5)
    val_q_index = np.random.choice(index, val_q_num, replace=False)
    test_q_index = list(set(index)- set(val_q_index))

    clearsky_q_rgb_train = [os.path.join(rgb_path[SCENE.Clearsky.value], f"{idx:06}") + '.png' for idx in train_q_index]
    clearsky_q_t_train = [os.path.join(thermal_path[SCENE.Clearsky.value], f"{idx:06}") + '.png' for idx in train_q_index]
    clearsky_q_pose_train = clearsky_pose_gt[train_q_index,:]
    print(f"- Created ClearSky Query \t: {len(clearsky_q_rgb_train)}")

    ## create rainy query
    rainy_pose_pd = rainy_pose_pd.iloc[1:-1, :].reset_index(drop=True)
    rainy_pose_gt = rainy_pose_pd.iloc[:, 1:3].to_numpy()
    rainy_index = [i for i in range(rainy_pose_gt.shape[0])]

    rainy_train_index = [0]
    rainy_train_pose = rainy_pose_gt[0, :].reshape(1, -1)  # add the first frame

    for i in range(1, rainy_pose_gt.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(rainy_train_pose[:, 0:2])
        dis, index = knn.kneighbors(rainy_pose_gt[i, 0:2].reshape(1, -1), 1, return_distance=True)

        if dis > Query_DIS_TH:
            rainy_train_pose = np.concatenate((rainy_train_pose, rainy_pose_gt[i, :].reshape(1, -1)), axis=0)
            rainy_train_index.append(i)

    train_q_index = rainy_train_index
    index = list(set(rainy_index)- set(train_q_index))
    
    val_q_num = int(len(index) *0.5)
    val_q_index = np.random.choice(index, val_q_num, replace=False)
    test_q_index = list(set(index)- set(val_q_index))

    rainy_q_rgb_train = [os.path.join(rgb_path[SCENE.Rainy.value], f"{idx:06}") + '.png' for idx in train_q_index]
    rainy_q_t_train = [os.path.join(thermal_path[SCENE.Rainy.value], f"{idx:06}") + '.png' for idx in train_q_index]
    rainy_q_pose_train = rainy_pose_gt[train_q_index,:]
    print(f"- Created Rainy Query \t: {len(rainy_q_rgb_train)}")

    ## create nighttime query
    nighttime_pose_pd = nighttime_pose_pd.iloc[1:-1, :].reset_index(drop=True)
    nighttime_pose_gt = nighttime_pose_pd.iloc[:, 1:3].to_numpy()
    nighttime_index = [i for i in range(nighttime_pose_gt.shape[0])]

    nighttime_train_index = [0]
    nighttime_train_pose = nighttime_pose_gt[0, :].reshape(1, -1)  # add the first frame

    for i in range(1, nighttime_pose_gt.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(nighttime_train_pose[:, 0:2])
        dis, index = knn.kneighbors(nighttime_pose_gt[i, 0:2].reshape(1, -1), 1, return_distance=True)

        if dis > Query_DIS_TH:
            nighttime_train_pose = np.concatenate((nighttime_train_pose, nighttime_pose_gt[i, :].reshape(1, -1)), axis=0)
            nighttime_train_index.append(i)

    train_q_index = nighttime_train_index
    index = list(set(nighttime_index)- set(train_q_index))
    
    val_q_num = int(len(index) *0.5)
    val_q_index = np.random.choice(index, val_q_num, replace=False)
    test_q_index = list(set(index)- set(val_q_index))

    nighttime_q_rgb_train = [os.path.join(rgb_path[SCENE.Nighttime.value], f"{idx:06}") + '.png' for idx in train_q_index]
    nighttime_q_t_train = [os.path.join(thermal_path[SCENE.Nighttime.value], f"{idx:06}") + '.png' for idx in train_q_index]
    nighttime_q_pose_train = nighttime_pose_gt[train_q_index,:]
    print(f"- Created NightTime Query \t: {len(nighttime_q_rgb_train)}")

    posDistThr = 18
    posDistSqThr = posDistThr**2
    nonTrivPosDistSqThr = 9**2
    
    # train ver
    dbStruct = {
        'whichSet': 'train',
        'db_rgb': morning_db_rgb,
        'db_t': morning_db_t,
        'db_pose': morning_db,
        'q_rgb_morning': morning_q_rgb_train,
        'q_t_morning': morning_q_t_train,
        'q_pose_morning': morning_q_pose_train,
        'q_rgb_clearsky': clearsky_q_rgb_train,
        'q_t_clearsky': clearsky_q_t_train,
        'q_pose_clearsky': clearsky_q_pose_train,
        'q_rgb_rainy': rainy_q_rgb_train,
        'q_t_rainy': rainy_q_t_train,
        'q_pose_rainy': rainy_q_pose_train,
        'q_rgb_nighttime': nighttime_q_rgb_train,
        'q_t_nighttime': nighttime_q_t_train,
        'q_pose_nighttime': nighttime_q_pose_train,
        'numDB': len(morning_db_rgb),
        'numQ_morning': len(morning_q_rgb_train),
        'numQ_clearsky': len(clearsky_q_rgb_train),
        'numQ_rainy': len(rainy_q_rgb_train),
        'numQ_nighttime': len(nighttime_q_rgb_train),
        'posDistThr': posDistThr,
        'posDistSqThr': posDistSqThr,
        'nonTrivPosDistSqThr': nonTrivPosDistSqThr
    }
    matfile = os.path.join(save_path_train, 'ms2_train.mat')
    io.savemat(matfile, {'dbStruct': dbStruct})

    # test ver
    dbStruct = {
        'whichSet': 'test',
        'db_rgb': morning_db_rgb,
        'db_t': morning_db_t,
        'db_pose': morning_db,
        'q_rgb_morning': morning_q_rgb_train,
        'q_t_morning': morning_q_t_train,
        'q_pose_morning': morning_q_pose_train,
        'q_rgb_clearsky': clearsky_q_rgb_train,
        'q_t_clearsky': clearsky_q_t_train,
        'q_pose_clearsky': clearsky_q_pose_train,
        'q_rgb_rainy': rainy_q_rgb_train,
        'q_t_rainy': rainy_q_t_train,
        'q_pose_rainy': rainy_q_pose_train,
        'q_rgb_nighttime': nighttime_q_rgb_train,
        'q_t_nighttime': nighttime_q_t_train,
        'q_pose_nighttime': nighttime_q_pose_train,
        'numDB': len(morning_db_rgb),
        'numQ_morning': len(morning_q_rgb_train),
        'numQ_clearsky': len(clearsky_q_rgb_train),
        'numQ_rainy': len(rainy_q_rgb_train),
        'numQ_nighttime': len(nighttime_q_rgb_train),
        'posDistThr': posDistThr,
        'posDistSqThr': posDistSqThr,
        'nonTrivPosDistSqThr': nonTrivPosDistSqThr
    }
    matfile = os.path.join(save_path_test, 'ms2_test.mat')
    io.savemat(matfile, {'dbStruct': dbStruct})    
    print("=" * 50)
