# save/load keyboard control sequences, images, and SLAM maps

import numpy as np
import cv2
import time
import csv
import os
import json
import map_eval_help

# for SLAM (M2), save the map
class OutputWriter:
    def __init__(self, folder_name="output/"):
        if not folder_name.endswith("/"):
            folder_name = folder_name + "/"
        self.folder = folder_name
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        self.map_f = folder_name+"slam.txt"
        self.targets_f = folder_name+"targets.txt"
        self.full_map_f = folder_name+"map_full.txt"
        self.image_count = 0
        
    
    def write_map(self, slam):
        map_dict = {"taglist":slam.taglist,
                    "map":slam.markers.tolist(),
                    "covariance":slam.P[3:,3:].tolist()}
        with open(self.map_f, 'w') as map_f:
            json.dump(map_dict, map_f, indent=2)
    
    def write_fruit_preds(self, fruit_list:dict):
        fruit_list_known = {}
        for (key, item) in fruit_list.items():
            fruit_list_known[key] = {'x': item['x'], 'y': item['y']}
        with open(self.targets_f, 'w') as fo:
            json.dump(fruit_list_known, fo, indent=4)

    def make_final_map(self):
        aruco_est = parse_slam_map(self.map_f)
        objects_est = parse_object_map(self.targets_f)
        
        total_est = {}
        for (key, item) in aruco_est.items():
            total_est[key] = item
        for (key, item) in objects_est.items():
            total_est[key] = item
        with open(self.full_map_f, 'w') as fo:
            json.dump(total_est, fo, indent=4)

    def compare_truth(self):
        fruit_dict_pred, aruco_dict_pred = parse_full_truth_map(self.full_map_f)
        fruit_dict_gt, aruco_dict_gt = parse_full_truth_map('home_true.txt')

        taglist, slam_est_vec, slam_gt_vec = map_eval_help.match_aruco_points(aruco_dict_pred, aruco_dict_gt)
        theta, x = map_eval_help.solve_umeyama2d(slam_est_vec, slam_gt_vec)
        slam_est_vec_aligned = map_eval_help.apply_transform(theta, x, slam_est_vec)
        slam_rmse_aligned = map_eval_help.compute_slam_rmse(slam_est_vec_aligned, slam_gt_vec)

        objects_est_aligned = map_eval_help.align_object_poses(theta, x, fruit_dict_pred)
        object_est_errors_aligned = map_eval_help.compute_object_est_error(fruit_dict_gt, objects_est_aligned)
        print('Object pose estimation errors after alignment:')
        print(json.dumps(object_est_errors_aligned, indent=4))

        print(f'The SLAM RMSE after alignment = {np.round(slam_rmse_aligned, 3)}')
        return
    
@staticmethod
def parse_object_map(fname):
    with open(fname, 'r') as fd:
        usr_dict = json.load(fd)
    target_dict = {}
    for key in usr_dict:
        target_dict[key] = {'x': usr_dict[key]['x'], 'y':usr_dict[key]['y']} 
    return target_dict

@staticmethod
def parse_slam_map(fname: str) -> dict:
    with open(fname, 'r') as fd:
        usr_dict = json.load(fd)
    aruco_dict = {}
    for (i, tag) in enumerate(usr_dict['taglist']):
        aruco_dict[f'aruco{tag}_0'] = {'x': usr_dict['map'][0][i], 'y':usr_dict['map'][1][i]}
    #print(f'Estimated marker poses: {aruco_dict}')
    return aruco_dict

def parse_full_truth_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_dict = {}
        aruco_dict = {}
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 3)
            y = np.round(gt_dict[key]['y'], 3)

            if key.startswith('aruco'):
                aruco_dict[key] = np.array([[x], [y]])
            else:
                fruit_dict[key] = np.array([[x, y]])
        return fruit_dict, aruco_dict
    
