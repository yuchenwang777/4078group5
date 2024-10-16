from util2.fruit_search_help import read_true_map, read_search_list, order_shopping_list
import util.measure as measure
import numpy as np
from TargetPoseEst import estimate_pose
from TargetPoseEst import merge_estimations
import os

class shopping_manager:

    def __init__(self, fruits_list=[], fruits_true_pos=[], shopping_list_path=None, wp_threshhold = 0.1):

        self.detected_fruit_type_occurances = []            # seen fruit (every detection)
        self.grocery_idx = 0
        self.shopping_list = []
        
        self.grocery_pos_list = fruits_true_pos
        self.grocery_name_list = fruits_list

        self.wp_distance_threshold = wp_threshhold
        
        # Find all fruit on shopping list
        for fruit_id, fruit_pos in zip(fruits_list, fruits_true_pos):
            self.shopping_list.append([fruit_id, tuple(fruit_pos)])

        # Order shopping list
        if shopping_list_path:
            shopping_order = read_search_list(shopping_list_path)
            #print(shopping_order, self.shopping_list)
            self.shopping_list = order_shopping_list(shopping_order, self.shopping_list)

        # Create total fruit dict
        self.fruit_pose_dict = {}
        for i in range(len(fruits_true_pos)):
            occurrence = self.detected_fruit_type_occurances.count(fruits_list[i])
            self.detected_fruit_type_occurances.append(fruits_list[i])
            target_pose =   {'y': fruits_true_pos[i][1],
                            'x': fruits_true_pos[i][0],
                            'known': True}
            self.fruit_pose_dict[f'{fruits_list[i]}_{occurrence}'] = target_pose
        

    def get_total_obstacles(self, obstacles:list):
        fruit_to_find_pos = self.shopping_list[self.grocery_idx][1]
        total_obstacles = []
        
        for i in range(len(obstacles[0])):
            total_obstacles.append((obstacles[0][i],obstacles[1][i]))

        for (fruit_id, fruit_pos) in self.fruit_pose_dict.items():
            # make sure goal fruit is not obstacle
            if (fruit_id.split("_")[0] != self.shopping_list[self.grocery_idx][0]):
                total_obstacles.append((fruit_pos['x'], fruit_pos['y']))
        return total_obstacles

    def get_fruit_to_find_pos(self):
        fruit_name = self.shopping_list[self.grocery_idx][0]
        for (fruit_id, fruit_pos) in self.fruit_pose_dict.items():
            if fruit_id.split("_")[0] == fruit_name:
                return (fruit_pos['x'], fruit_pos['y'])
        print("fruit not on shopping list ?")
        # get from truth map
        return self.shopping_list[self.grocery_idx][1]

    def is_not_finished(self):
        return self.grocery_idx < len(self.shopping_list)
    
    def next_fruit(self):
        self.grocery_idx += 1

    def reset(self):
        self.grocery_idx = 0
        self.detected_fruit_type_occurances = []
        self.grocery_idx = 0
        self.shopping_list = []
        self.fruit_pose_dict = {}

    def consider_detected_fruit(self, detection, robot_state):
        # Detected fruit logic
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        fileK = f'{script_dir}/calibration/param/intrinsic.txt'
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        occurrence = self.detected_fruit_type_occurances.count(detection[0])
        abs_dist, rel_dist = estimate_pose(camera_matrix, detection, robot_state)
        valid_detection = False
        b_update_wp_distance = False
        b_found_new_fruit = False
        if (abs_dist is not None) and (rel_dist is not None):
            if rel_dist["x"] < 0.7:
                valid_detection = True
                self.fruit_pose_dict[f'{detection[0]}_{occurrence}'] = abs_dist
                self.detected_fruit_type_occurances.append(detection[0])
                self.fruit_pose_dict, b_update_wp_distance_sample, b_merged = merge_estimations(self.fruit_pose_dict, self.wp_distance_threshold)
                if b_update_wp_distance_sample:
                    b_update_wp_distance = True
                if not b_update_wp_distance_sample and not b_merged:
                    b_update_wp_distance = True

        #print(self.fruit_pose_dict)

        return b_found_new_fruit, valid_detection, b_update_wp_distance