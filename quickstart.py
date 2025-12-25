# Generated from: quickstart.ipynb
# Converted at: 2025-12-22T08:15:17.517Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import cv2
import faiss
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import commons
import network
import sys
from Parser import Parser
from collections import OrderedDict

import network
import utils
import datasets_dual
from tqdm import tqdm
import os

parser = Parser()
sys.argv = ['a'] # 非常神奇，居然可直接这样模拟
args = parser.parse_arguments()
args.device = 'cuda'
print(args)

features_dim = 768
save_path = '/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/Dataset/save_mat'

# # Load the Model and Dataset


model = network.RGBTVPRNet(forward_fn = args.fuse_fn, pretrained_foundation = True, foundation_model_path = '/path/to/your/pretrained/weights')

model = model.to('cuda')

model = utils.resume_model('/path/to/your/checkpoint', model)
model = model.eval()

res = faiss.StandardGpuResources()
faiss_index_cpu = faiss.IndexFlatL2(features_dim)

DATASET_FOLDER = "/path/to/your/dataset"
args.img_time = 'allday'
args.sequences = ['SNU']
args.soft_positives_dist_threshold = 10

test_ds = datasets_dual.BaseSTheReODual(args, DATASET_FOLDER, split='test')
print(f"database: {test_ds.database_num}, queries: {test_ds.queries_num}")

db_ds = Subset(test_ds, list(range(test_ds.database_num)))
q_ds = Subset(test_ds, list(range(test_ds.database_num, len(test_ds))))

# # Compute the Descriptor


# Compute database features

database_dataloader = DataLoader(dataset=db_ds, num_workers=71,
                                        batch_size=196, pin_memory=True,
                                        shuffle=False)
# print(features_dim)
database_features = np.empty((test_ds.database_num, features_dim), dtype="float32")

with torch.no_grad():
    for inputs, indices in tqdm(database_dataloader):
        features = model(inputs.to('cuda')).view(-1, features_dim)
        features = features.cpu().numpy()
        # print(features.shape)
        database_features[indices.numpy(), :] = features

    # faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index_cpu)
    faiss_index = faiss_index_cpu
    faiss_index.add(database_features)
    # del database_features

# Compute query feature

query_index = 420   # specify the query index

with torch.no_grad():
    query, _ = q_ds[query_index]
    query = query.unsqueeze(0)
    query_feature = model(query.to('cuda')).cpu().numpy()
    
    positives_nums = 20 # retrieve 20 database samples
    distances, predictions = faiss_index.search(query_feature, positives_nums)


positives_indexes = test_ds.get_positives()[query_index]

print(f"positives: {positives_indexes}")
print(f"predictions: {predictions}")

if predictions[0][0] in positives_indexes:
    print("hit")
else:
    print("miss")

# find false positives
false_positives = []
for index in predictions[0]:
    if index not in positives_indexes:
        false_positives.append(index)

# save results
database_num = test_ds.database_num
query_rgb_path = test_ds.rgb_img_paths[query_index+database_num]
query_t_path = test_ds.t_img_paths[query_index+database_num]
query_pose = test_ds.queries_utms[query_index]

neighbors = []
for neighbor_index in predictions[0]:
    assert neighbor_index < database_num
    neighbor_rgb_path = test_ds.rgb_img_paths[neighbor_index]
    neighbor_t_path = test_ds.t_img_paths[neighbor_index]
    neighbor_pose = test_ds.database_utms[neighbor_index]

    neighbors.append({
        "rgb_path": neighbor_rgb_path,
        "t_path": neighbor_t_path,
        "pose": neighbor_pose
    })

results = {
            "query": {
                "rgb_path": query_rgb_path,
                "t_path": query_t_path,
                "pose": query_pose
            },
            "neighbors": neighbors
        }

# # Visualizing the retrieved image pairs


def read_rgb_img(path):
    rgb_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BAYER_BG2RGB)
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BAYER_RG2RGB)
    
    return rgb_image

def read_t_img(path):
    t_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    t_img = cv2.normalize(t_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
    return t_img

# Visualizing the retrieved image pairs
k = 5

fig, axes = plt.subplots(2, k + 1, figsize=(15, 5))
fig.suptitle(f"Query-{query_index} and Top-{k} Predictions", fontsize=24)
plt.subplots_adjust(top=0.5)
for ax in axes.flatten():
    ax.axis("off")
plt.subplots_adjust(hspace=-1.18)
plt.rcParams.update({'font.size': 16})

# show query image
query_rgb_img = read_rgb_img(results["query"]["rgb_path"])
query_t_img = read_t_img(results["query"]["t_path"])
axes[0, 0].imshow(query_rgb_img)
axes[0, 0].set_title("Query")
axes[1, 0].imshow(query_t_img)
# save query image
os.makedirs(f"{save_path}/q_{query_index}", exist_ok=True)
cv2.imwrite(f"{save_path}/q_{query_index}/q_rgb.png", query_rgb_img)
cv2.imwrite(f"{save_path}/q_{query_index}/q_t.png", query_t_img)

# show retrieved images
for i, neighbor in enumerate(results["neighbors"][:k]):
    rgb_img = read_rgb_img(neighbor["rgb_path"])
    t_img = read_t_img(neighbor["t_path"])
    axes[0, i + 1].imshow(rgb_img)
    axes[0, i + 1].set_title(f"predicitons {i + 1}")
    axes[1, i + 1].imshow(t_img)
    # save retrieved images
    cv2.imwrite(f"{save_path}/q_{query_index}/pr_{i + 1}_rgb.png", rgb_img)
    cv2.imwrite(f"{save_path}/q_{query_index}/pr_{i + 1}_t.png", t_img)

plt.tight_layout()
plt.show()