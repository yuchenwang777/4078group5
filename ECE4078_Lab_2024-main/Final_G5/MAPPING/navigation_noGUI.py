# teleoperate the robot, perform SLAM and object detection

import os
import sys
import time
import cv2
import numpy as np

# import utility functions
sys.path.insert(0, "{}/util2".format(os.getcwd()))
from util2.pibot import PenguinPi    # access the robot
import util2.DatasetHandler as dh    # save/load functions
import util2.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations
import argparse

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam2.ekf import EKF
from slam2.robot import Robot
import slam2.aruco_detector as aruco

# import YOLO components 
from YOLO.detector import Detector
from util2.clear_out import clear


import util2.measure as measure

from waypoint_manager import wp_manager
from util2.get_test_img import get_image
import copy
from util2.get_path_alg import path_alg
from shopping_manager import shopping_manager
from util2.fruit_search_help import read_true_map
import warnings
from util2.colors import colors

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

class Operate:

    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ekf_threshhold)
        
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers

        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'make_prediction': False,
                        'auto_fruit_search': False,
                        'click_map': False,
                        'compare_truth': False
                        }
        
        self.quit = False
        self.file_output = None
        self.ekf_on = False
        self.image_id = 0
        self.notification = 'Coordinates'
        self.pred_notifier = False

        if args.online.upper() == "TRUE":
            self.online = True  
        else: 
            self.online = False 

        # Timers 
        self.start_time = time.time()
        self.control_clock = time.time()

        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)

        self.detector_output = None
        self.detector = Detector(f'models/{args.yolo_model}')
        self.yolo_vis = np.ones((240, 320, 3)) * 100

        self.speed = 1
        self.last_action = "None"

        self.offline_img_no = args.offline_id

        self.ekf_on = False
        self.b_auto_fruit_search = False
        self.shopping_manager = shopping_manager()

        if args.path_alg.upper()[0] == "A":
            self.path_alg = "A*"
        elif args.path_alg.upper()[0] == "D":
            self.path_alg = "Dijkstra"
        elif args.path_alg.upper()[0] == "T":
            self.path_alg = "Theta*"

        self.obstacles_hazard = float(args.hazard_radius)
        self.a_star_res = float(args.path_res)
        self.hazard_boarder = float(args.hazard_boarder)
        self.a_star_delay = 500
        self.a_star_turning_delay = args.wp_random_update_interval
        self.a_star_timer = 0
        self.a_star_timer2 = time.time()
        self.shopping_list_f = args.shopping_list
        self.b_saw_law_last_tick = False
        self.colors = colors()
        self.color_mode = 2
        self.fruit_wp_threshhold = args.fruit_wp_threshhold
    
    def start_search(self):
        if not self.b_auto_fruit_search:
            print('fruit search')
            self.ekf_on = True
            self.command['inference'] = True
            self.b_auto_fruit_search = True
            self.init_fruit_search()
            self.notification = 'Shopping'
            

    # wheel control
    def control(self):
        if self.online:
            drive_meas = self.pibot.set_velocity(self.command['motion'])
        else:
            drive_meas = self.pibot.set_velocity([0, 0])
        return drive_meas 

    # camera control
    def take_pic(self):
        if self.online:
            self.img = self.pibot.get_image()
        else:
            self.img = get_image(self.offline_img_no)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img, rvecs, tvecs, ids = self.aruco_det.detect_marker_positions(self.img)

        if not self.ekf_on:
            self.ekf_on = True
        
        lms = self.ekf.remove_unknown_lms(lms)
        lms = self.ekf.get_centre_pos(lms, rvecs, tvecs, ids)
        lms = self.ekf.change_landmark_covar(lms)

        if len(lms) > 0:
            self.ekf.add_landmarks(lms)
        else:
            self.ekf.predict(drive_meas)

        b_state_change_threshhold, average = self.ekf.update(lms)
        
        return average
    
    # Yolo detect
    def detect_target(self):
        if not self.command['inference'] or self.detector is None:
            return
        
        if time.time() - self.yolo_timer < self.yolo_delay:
            return
        self.yolo_timer = time.time()

        yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)
        self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)
        self.file_output = (yolo_input_img, self.ekf)
        if self.b_auto_fruit_search:
            self.auto_fruit_search()

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.save_slam_state(self.ekf)
        # save inference with the matching robot pose and detector labels
        if self.command['make_prediction']:
            self.output.save_inference(self.file_output)
 
    # Compare with Ground Truth (for testing)
    def compare_truth(self):
        if not self.command['compare_truth']:
            return
        
        print("COMPARE TRUTH")
        self.output.compare_truth()
        self.command['compare_truth'] = False

    # Get Control inputs for auto search behavior
    def auto_fruit_search(self):
        if self.b_auto_fruit_search:
            self.update_wps()

    # Update Path waypoints
    def update_wps(self):
        if not self.b_auto_fruit_search:
            return
            
        if self.wp_manager.arrived:
            return
            
        if self.shopping_manager.grocery_idx >= len(self.shopping_manager.shopping_list):
            return
            
        total_obstacles = self.shopping_manager.get_total_obstacles(copy.deepcopy(self.ekf.markers))
        fruit_to_find_pos = self.shopping_manager.get_fruit_to_find_pos()
        new_wps, end_pos = path_alg(self, self.path_alg, fruit_to_find_pos, total_obstacles, self.colors)
        if len(new_wps) == 0:
            return
        else:
            self.wp_manager.reset(new_wps, end_pos)
            self.wp_changed_last_iter = True
    
    # Init EKF
    def init_ekf(self, datadir, ekf_threshhold):  
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot, ekf_threshhold)

    # Init auto fruit search
    def init_fruit_search(self):
        self.a_star_timer = time.time()
        # Get ground truths
        fruits_list, fruits_true_pos, aruco_true_pos = read_true_map('lab_output/points.txt')
        lms = []
        for i in range(len(aruco_true_pos)):
            lms.append([aruco_true_pos[i][0], aruco_true_pos[i][1], 0])
        self.ekf.add_landmarks(lms, known_lm=True)
        
        self.shopping_manager = shopping_manager(fruits_list, fruits_true_pos, self.shopping_list_f, self.fruit_wp_threshhold)

        self.path_found = True
        self.update_wps()

    # Get Real Coordinates of Mouse click
    def output_coordinate_click(self, ekf_view_pos, ekf_view_size):
        if not self.command['click_map']:
            return
        ekf_view_x, ekf_view_y = ekf_view_pos
        ekf_view_w, ekf_view_h = ekf_view_size
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_x, mouse_y = mouse_x - ekf_view_x, mouse_y - ekf_view_y
        real_x, real_y = EKF.to_xy_coor((mouse_x, mouse_y),(ekf_view_w,ekf_view_h), self.ekf.m2pixel)
        robot_xy = self.ekf.robot.state[:2, 0].reshape((2, 1))
        pos_x, pos_y = real_x + robot_xy[0], real_y + robot_xy[1]
        self.notification = f'x: {np.round(pos_x, 2)}, y: {np.round(pos_y, 2)}'
        self.command['click_map'] = False
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--yolo_model", default='yolov8_model.pt')
    parser.add_argument("--offline_id", default='')

    parser.add_argument("--online", type=str, default="True")
    parser.add_argument("--map", type=str)
    parser.add_argument("--shopping_list",  type=str, default='shopping_list.txt')
    parser.add_argument("--hazard_radius", type=str, default='0.2')
    parser.add_argument("--path_res", type=str, default='0.1')
    parser.add_argument("--hazard_boarder", type=str, default='1')
    parser.add_argument("--path_alg", type=str, default='t')

    parser.add_argument("--ekf_threshhold", type=float, default=0.001)
    parser.add_argument("--fruit_wp_threshhold", type=float, default=0.1)
    parser.add_argument("--wp_random_update_interval", type=float, default=300)
    parser.add_argument("--yolo_delay", type=float, default=0.5)

    args, _ = parser.parse_known_args()

    operate = Operate(args)
    start = True
    save_time = time.time()
    
    while start:
        if not operate.quit:
            #operate.update_keyboard()
            operate.take_pic()
            drive_meas = operate.control()
            operate.update_slam(drive_meas)
            operate.detect_target()
            operate.record_data()
            operate.compare_truth()
            operate.auto_fruit_search()
            operate.start_search()
        else:
            break