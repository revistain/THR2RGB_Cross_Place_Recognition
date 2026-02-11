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

np.random.seed(42)

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
dataset_path = [os.path.join(dataset_dir, index) for index in timestamp_list]    
pose_path = [pose_dir + f'/{path}_timestamp_UTM.csv' for path in timestamp_list]  
rgb_path = [path + '/rgb/img_left/' for path in dataset_path]   
thermal_path = [path + '/thr/img_left/' for path in dataset_path]   

morning_pose_pd = pd.read_csv(pose_path[0], header=None)
afternoon_pose_pd = pd.read_csv(pose_path[1], header=None)
evening_pose_pd = pd.read_csv(pose_path[2], header=None)

DB_DIS_TH = 5
Query_DIS_TH = 1

save_path = os.path.join('save_mat', 'train', 'ms2_train')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"create {save_path} ")

morning_pose_pd = morning_pose_pd.iloc[1:-1, :].reset_index(drop=True)
morning_pose_gt = morning_pose_pd.iloc[:, 1:3].to_numpy()
# print(morning_pose_gt.shape)

## create morning database

db_index = [0]
morning_db = morning_pose_gt[0, :].reshape(1, -1)  # add the first frame

for i in range(1, morning_pose_gt.shape[0]):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(morning_db[:, 0:2])
    dis, index = knn.kneighbors(morning_pose_gt[i, 0:2].reshape(1, -1), 1, return_distance=True)

    if dis > DB_DIS_TH:
        morning_db = np.concatenate((morning_db, morning_pose_gt[i, :].reshape(1, -1)), axis=0)
        db_index.append(i)

print(f"morning dataset has {morning_pose_gt.shape[0]} poses")
print(f"morning database has {morning_db.shape[0]} poses")

morning_rgb_list = []
for filename in os.listdir(rgb_path[0]):
    if filename.endswith('.png'):
        name_without_extension = os.path.splitext(filename)[0]
        morning_rgb_list.append(name_without_extension)

morning_t_list = []
for filename in os.listdir(thermal_path[0]):
    if filename.endswith('.png'):
        name_without_extension = os.path.splitext(filename)[0]
        morning_t_list.append(name_without_extension)

morning_rgb_list.sort()
morning_t_list.sort()

morning_rgb_time = np.array(morning_rgb_list).astype(np.int64)* 10e-10  
morning_t_time = np.array(morning_t_list).astype(np.int64)* 10e-10
morning_db_time = morning_pose_pd.iloc[db_index, 0].to_numpy()

distances = np.abs(morning_rgb_time[:, np.newaxis] - morning_db_time)
morning_db_rgb_index = np.argmin(distances, axis=0)
morning_db_rgb = [rgb_path[0] + morning_rgb_list[i] + '.png' for i in morning_db_rgb_index]

distances = np.abs(morning_t_time[:, np.newaxis] - morning_db_time)
morning_db_t_index = np.argmin(distances, axis=0)
morning_db_t = [thermal_path[0] + morning_t_list[i] + '.png' for i in morning_db_t_index]

print(len(morning_db_rgb), len(morning_db_t), morning_db.shape[0])
assert len(morning_db_rgb) == len(morning_db_t) == morning_db.shape[0]
# ## Visualization
# random_index = np.random.randint(0, len(morning_db_t))

# img1 = cv2.imread(morning_db_rgb[random_index], cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(dataset_dir + morning_db_t[random_index], cv2.IMREAD_UNCHANGED)
# # img2 = cv2.imread(morning_db_rgb[random_index+10], cv2.IMREAD_GRAYSCALE)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BAYER_RG2RGB)
# img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# # img2 = cv2.cvtColor(img2, cv2.COLOR_BAYER_RG2RGB)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(img1)
# ax1.axis('off')  
# ax1.set_title('Database rgb')
# ax2.imshow(img2, cmap='gray')
# ax2.axis('off')  
# ax2.set_title('Database T')
# plt.tight_layout()
# plt.show()

## create morning query
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

# get train image
morning_query_time = morning_pose_pd.iloc[train_q_index, 0].to_numpy()
distances = np.abs(morning_rgb_time[:, np.newaxis] - morning_query_time)
morning_q_rgb_index = np.argmin(distances, axis=0)
morning_q_rgb_train = [os.path.join(rgb_path[0], morning_rgb_list[i]) + '.png' for i in morning_q_rgb_index]

distances = np.abs(morning_t_time[:, np.newaxis] - morning_query_time)
morning_q_t_index = np.argmin(distances, axis=0)
morning_q_t_train = [os.path.join(thermal_path[0], morning_t_list[i]) + '.png' for i in morning_q_t_index]

morning_q_pose_train = morning_pose_gt[train_q_index,:]

posDistThr = 18
posDistSqThr = posDistThr**2
nonTrivPosDistSqThr = 9**2
# posDistThr, posDistSqThr, nonTrivPosDistSqThr are not used

dbStruct = {
    'whichSet': 'train',
    'db_rgb': morning_db_rgb,
    'db_t': morning_db_t,
    'db_pose': morning_db,
    'q_rgb_morning': morning_q_rgb_train,
    'q_t_morning': morning_q_t_train,
    'q_pose_morning': morning_q_pose_train,
    'q_rgb_afternoon': afternoon_q_rgb_train,
    'q_t_afternoon': afternoon_q_t_train,
    'q_pose_afternoon': afternoon_q_pose_train,
    'q_rgb_evening': evening_q_rgb_train,
    'q_t_evening': evening_q_t_train,
    'q_pose_evening': evening_q_pose_train,
    'numDB': len(morning_db_rgb),
    'numQ_morning': len(morning_q_rgb_train),
    'numQ_afternoon': len(afternoon_q_rgb_train),
    'numQ_evening': len(evening_q_rgb_train),
    'posDistThr': posDistThr,
    'posDistSqThr': posDistSqThr,
    'nonTrivPosDistSqThr': nonTrivPosDistSqThr
}
matfile = os.path.join(save_path,  'sthereo_train.mat')
io.savemat(matfile, {'dbStruct': dbStruct})

dbStruct = {
    'whichSet': 'val',
    'db_rgb': morning_db_rgb,
    'db_t': morning_db_t,
    'db_pose': morning_db,
    'q_rgb_morning': morning_q_rgb_val,
    'q_t_morning': morning_q_t_val,
    'q_pose_morning': morning_q_pose_val,
    'q_rgb_afternoon': afternoon_q_rgb_val,
    'q_t_afternoon': afternoon_q_t_val,
    'q_pose_afternoon': afternoon_q_pose_val,
    'q_rgb_evening': evening_q_rgb_val,
    'q_t_evening': evening_q_t_val,
    'q_pose_evening': evening_q_pose_val,
    'numDB': len(morning_db_rgb),
    'numQ_morning': len(morning_q_rgb_val),
    'numQ_afternoon': len(afternoon_q_rgb_val),
    'numQ_evening': len(evening_q_rgb_val),
    'posDistThr': posDistThr,
    'posDistSqThr': posDistSqThr,
    'nonTrivPosDistSqThr': nonTrivPosDistSqThr
}
matfile = os.path.join(save_path,  'sthereo_val.mat')
io.savemat(matfile, {'dbStruct': dbStruct})

dbStruct = {
    'whichSet': 'test',
    'db_rgb': morning_db_rgb,
    'db_t': morning_db_t,
    'db_pose': morning_db,
    'q_rgb_morning': morning_q_rgb_test,
    'q_t_morning': morning_q_t_test,
    'q_pose_morning': morning_q_pose_test,
    'q_rgb_afternoon': afternoon_q_rgb_test,
    'q_t_afternoon': afternoon_q_t_test,
    'q_pose_afternoon': afternoon_q_pose_test,
    'q_rgb_evening': evening_q_rgb_test,
    'q_t_evening': evening_q_t_test,
    'q_pose_evening': evening_q_pose_test,
    'numDB': len(morning_db_rgb),
    'numQ_morning': len(morning_q_rgb_test),
    'numQ_afternoon': len(afternoon_q_rgb_test),
    'numQ_evening': len(evening_q_rgb_test),
    'posDistThr': posDistThr,
    'posDistSqThr': posDistSqThr,
    'nonTrivPosDistSqThr': nonTrivPosDistSqThr
}
matfile = os.path.join(save_path,  'sthereo_test.mat')
io.savemat(matfile, {'dbStruct': dbStruct})