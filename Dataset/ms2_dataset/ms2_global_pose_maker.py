import os
import numpy as np
import pandas as pd
from pyproj import Transformer

np.random.seed(42)
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

ROOT_DIR = '/data2/datasets/sync_data'
if __name__ == "__main__":
    for scene_name in timestamp_list:
        print(f" ===== Iterating Scene: {scene_name} =====")
        scene_root_path = os.path.join(ROOT_DIR, scene_name)
        
        # 1. 먼저 gps_timestamp를 가져오기 + 공간 잡아두기
        timestamp_UTM = []
        scene_gps_ts_path = os.path.join(scene_root_path, 'gps_imu', 'data_timestamp.txt')
        with open(scene_gps_ts_path, "r") as gps_ts_file:
            _gps_timestamps = gps_ts_file.readlines()
            for gps_timestamp in _gps_timestamps:
                timestamp_UTM.append([int(gps_timestamp.strip()), 0.0, 0.0])
        
        # 2. GPS도 가져오기 + GPS를 UTM(Easting, Northing)으로 변환
        scene_gps_path = os.path.join(scene_root_path, 'gps_imu', 'data')
        gps_paths = sorted(os.listdir(scene_gps_path))
        gps2utm = Transformer.from_crs("epsg:4326", "epsg:32652", always_xy=True)
        for idx, gps_path_name in enumerate(gps_paths):
            gps_path = os.path.join(scene_gps_path, gps_path_name)
        
            with open(gps_path, "r") as gps_file:
                gps_lines = gps_file.readlines()
                easting, northing = gps2utm.transform(float(gps_lines[0].strip()), float(gps_lines[2].strip()))
                timestamp_UTM[idx][1] = easting
                timestamp_UTM[idx][2] = northing
        
        # 3. list2csv
        pd.DataFrame(timestamp_UTM).to_csv(f'{scene_name}_timestamp_UTM.csv', header=False, index=False)