import os

ROOT_PATH = "/data2/datasets/sync_data"
dirs = os.listdir(ROOT_PATH)
dirs = [_ for _ in dirs if _.startswith("_2021")]
for dir in sorted(dirs):
    image_dir1 = os.listdir(os.path.join(ROOT_PATH, dir, "rgb", "img_left"))
    image_dir2 = os.listdir(os.path.join(ROOT_PATH, dir, "rgb", "img_right"))
    image_dir3 = os.listdir(os.path.join(ROOT_PATH, dir, "thr", "img_left"))
    image_dir4 = os.listdir(os.path.join(ROOT_PATH, dir, "thr", "img_right"))
    
    print(f" ==== Path: {dir} ====")
    print(f"rgb_left: {len(image_dir1)}")
    print(f"rgb_right: {len(image_dir2)}")
    print(f"thr_left: {len(image_dir3)}")
    print(f"thr_right: {len(image_dir4)}")