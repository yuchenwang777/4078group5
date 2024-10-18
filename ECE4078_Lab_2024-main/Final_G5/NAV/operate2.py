# teleoperate the robot, perform SLAM and object detection

import os
import sys
import time
import cv2
import numpy as np
from gauss import get_gaussian_value_map



# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations

# Set SDL to use the dummy NULL video driver, so it doesn't need a windowing system.
#os.environ["SDL_VIDEODRIVER"] = "dummy"

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import YOLO components 
from detector import Detector
from util.clear_out import clear


import util.measure as measure

from waypoint_manager import wp_manager
from util.get_test_img import get_image
import copy
from util.get_path_alg import path_alg
from shopping_manager import shopping_manager
from util.fruit_search_help import read_true_map
import warnings
from util.colors import colors

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

class Operate:

    def __init__(self, args, canvas):
        

        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ekf_threshhold)
        self.custome_map= None
        
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
        self.count_down = 600

        # Timers 
        self.start_time = time.time()
        self.control_clock = time.time()

        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)

        self.detector_output = None
        self.detector = Detector(f'models/{args.yolo_model}.pt')
        self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.BACKGROUND_IMG = pygame.image.load('pics/gui_mask.jpg')
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        self.speed = 1
        self.last_action = "None"

        self.offline_img_no = args.offline_id

        self.v_pad = 40
        self.h_pad = 20

        self.gui_width = 1400 # prev: 700, 660
        self.gui_height = 600

        self.ekf_view_width = 650
        self.ekf_view_height = 610

        self.caption_v_pad = 10

        self.command_w = 170
        self.command_h = 220
        

        self.canvas = canvas
        self.wp_changed_last_iter = False

        if args.online.upper() == "TRUE":
            self.online = True  
        else: 
            self.online = False  

        self.wp_manager = wp_manager()
        
        self.yolo_delay = args.yolo_delay
        self.yolo_timer = time.time()

        self.ekf_on = False
        self.b_auto_fruit_search = False
        self.shopping_manager = shopping_manager()

        if args.path_alg.upper()[0] == "A":
            self.path_alg = "A*"
        elif args.path_alg.upper()[0] == "D":
            self.path_alg = "Dijkstra"
        elif args.path_alg.upper()[0] == "T":
            self.path_alg = "A* Tom"
        self.obstacles_hazard = float(args.hazard_radius)         
        self.a_star_res = float(args.path_res)                    
        self.hazard_boarder = float(args.hazard_boarder)          
        # Start YOLO model and set up callback timer
        self.a_star_delay = 500
        self.a_star_turning_delay = args.wp_random_update_interval
        self.a_star_timer = 0
        self.a_star_timer2 = time.time()
        self.shopping_list_f = args.shopping_list
        self.b_saw_law_last_tick = False
        self.colors = colors()
        self.color_mode = 2
        self.fruit_wp_threshhold = args.fruit_wp_threshhold

    # Function to control robot wheels
    def control_robot(self):
        if self.online:
            # Send velocity commands to robot if in online mode
            lv, rv = self.pibot.set_velocity(self.command['motion'])
            dt = time.time() - self.control_clock  # Time delta since last command
            drive_meas = measure.Drive(lv, -rv, dt)  # Record wheel velocities
        else:
            # If in offline mode, simulate stationary robot
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(0, 0, dt)
        self.control_clock = time.time()  # Update control clock
        return drive_meas

    # Capture an image from the robot's camera
    def take_pic(self):
        if self.online:
            # Get a live image from the robot's camera
            self.img = self.pibot.get_image()
        else:
            # Load an offline image for testing
            self.img = get_image(f'offline_img{self.offline_img_no}')

    # SLAM (Simultaneous Localization and Mapping) with ArUco markers
    def update_slam(self, drive_meas):
        # Detect ArUco markers in the current image
        lms, self.aruco_img, rvecs, tvecs, ids = self.aruco_det.detect_marker_positions(self.img)

        if not self.ekf_on:
            return None  # Return if SLAM is turned off

        # Filter out unknown landmarks and adjust for the robot's position
        lms = self.ekf.remove_unknown_lms(lms)
        lms = self.ekf.get_centre_pos(lms, rvecs, tvecs, ids)
        lms = self.ekf.change_landmark_covar(lms)

        if len(lms) > 0:
            self.b_saw_law_last_tick = True  # Saw landmarks
        else:
            self.b_saw_law_last_tick = False  # No landmarks detected

        # Run SLAM algorithm
        self.ekf.predict(drive_meas)  # Predict robot's next state
        self.ekf.add_landmarks(lms)   # Add detected landmarks to map
        b_state_change_threshhold, average = self.ekf.update(lms)  # Update SLAM state
        
        return average  # Return SLAM update average for analysis

    
    # Yolo detect
    def detect_target(self):

        # If inference command is not active or the detector is not initialized, exit the function
        if not self.command['inference'] or self.detector is None:
            return
        
        # Check if enough time has passed since the last YOLO detection
        if time.time() - self.yolo_timer < self.yolo_delay:
            return
        # Update the timer for the next detection
        self.yolo_timer = time.time()

        # Convert the input image from RGB to BGR (as YOLO expects)
        yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        # Perform detection on the input image and save the visualization
        self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)
        self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)
        self.file_output = (yolo_input_img, self.ekf)

        # If automatic fruit search is enabled, process the detected fruits
        if self.b_auto_fruit_search:
            if self.detector_output:
                # Consider each detected fruit and update waypoints or goals accordingly
                for detection in self.detector_output:
                    b_found_new_fruit, valid_detection, b_update_wp_sample = self.shopping_manager.consider_detected_fruit(detection, self.ekf.robot.state)

    # Save SLAM map and predictions
    def record_data(self):
        # If output command is active, save SLAM data and fruit predictions
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.output.write_fruit_preds(self.shopping_manager.fruit_pose_dict)
            self.output.make_final_map()
            self.notification = 'Map is saved'
            self.command['output'] = False

        # Save predictions with the robot pose and detected fruits if commanded
        if self.command['make_prediction']:
            if self.detector_output:
                for detection in self.detector_output:
                    b_found_new_fruit, valid_detection, b_update_wp_sample = self.shopping_manager.consider_detected_fruit(detection, self.ekf.robot.state)
                    if valid_detection:
                        self.output.write_fruit_preds(self.shopping_manager.fruit_pose_dict)
                        self.notification = f'Fruit Preds updated'
            else:
                # Notify if no predictions were available
                self.notification = f'No prediction in buffer, save ignored'
            self.command['make_prediction'] = False

    # Get control inputs for automatic fruit search behavior
    def perform_auto_fruit_search(self):
        if self.b_auto_fruit_search:

            # Update motion commands based on waypoint manager and SLAM data
            self.command['motion'] = self.wp_manager.tick(self.ekf.robot, self.path_found, self.b_saw_law_last_tick, self.ekf.markers)

            # Check if the current route is completed and there are more fruits to search
            if self.wp_manager.completed_route and self.shopping_manager.is_not_finished():
                self.shopping_manager.next_fruit()  # Move to the next fruit
                if self.shopping_manager.is_not_finished():
                    self.wp_manager.finished_waiting = False
                    print("UPDATE go next fruit")
                    self.update_waypoints()  # Update waypoints for the next fruit

    # Update waypoints for the path
    def update_waypoints(self):
        
        # If auto fruit search is not enabled, do nothing
        if not self.b_auto_fruit_search:
            return
        
        # If we have arrived at the current goal, do nothing
        if self.wp_manager.arrived:
            return
        
        # If we have searched all fruits, do nothing
        if self.shopping_manager.grocery_idx >= len(self.shopping_manager.shopping_list):
            return 
        
        # Get obstacles and the next fruit's position
        total_obstacles = self.shopping_manager.get_total_obstacles(copy.deepcopy(self.ekf.markers))
        fruit_to_find_pos = self.shopping_manager.get_fruit_to_find_pos()
        
        # Use the path algorithm to compute new waypoints to the fruit, considering obstacles
        new_wps, end_pos = path_alg(self, self.path_alg, fruit_to_find_pos, total_obstacles, self.colors, self.custom_map)

        # Check if the path was found and update the waypoint manager
        if len(new_wps) == 0:
            self.path_found = False
        else:
            self.path_found = True

        self.wp_manager.reset(new_wps, end_pos)
        self.wp_changed_last_iter = True

    # Initialize the EKF (Extended Kalman Filter)
    def init_ekf(self, datadir, ekf_threshhold):  
        # Load intrinsic and distortion parameters for the camera
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        
        # Load scale and baseline parameters for the robot's stereo vision
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        
        # Initialize the robot object and EKF with the loaded parameters
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot, ekf_threshhold)

    # Initialize automatic fruit search
    def init_fruit_search(self):

        self.a_star_timer = time.time()  # Timer for A* pathfinding algorithm
        # Get ground truths from the provided map file
        fruits_list, fruits_true_pos, aruco_true_pos = read_true_map('lab_output/map_full.txt')
        lms = []

        # Add true ArUco marker positions as landmarks
        for i in range(len(aruco_true_pos)):
            lmxy = [[aruco_true_pos[i][0]], [aruco_true_pos[i][1]]]
            lm = measure.Marker(position=lmxy, tag=i+1, covariance=(0.0*np.eye(2)))
            lms.append(lm)
        
        self.ekf.add_landmarks(lms, known_lm=True)

        # Initialize the shopping manager with the list of fruits and their positions
        self.shopping_manager = shopping_manager(fruits_list, fruits_true_pos, self.shopping_list_f, self.fruit_wp_threshhold)
        self.total_obstales = self.shopping_manager.get_total_obstacles(copy.deepcopy(self.ekf.markers))
        self.custom_map = get_gaussian_value_map(self.total_obstales, 19, 20, 3, 0.01) # put function later

        # Set the initial path state and update waypoints
        self.path_found = True
        self.update_waypoints()


    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,
                                position=(h_pad, 240 + 2 * v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
 
    
    def put_caption(self, canvas, caption, position):
        caption_surface = TITLE_FONT.render(caption,
                                            False, self.colors.COLOR_HEADING_TEXT)
        canvas.blit(caption_surface, (position[0], position[1] - 25))
    
    # keyboard teleoperation, replace with your M1 codes if preferred        
    def update_keyboard(self):
        for event in pygame.event.get():
            
            ######################################################################################
            ######################################################################################
            ### MANUAL MOTION
            ######################################################################################
            ######################################################################################
            if not self.b_auto_fruit_search:
                # press shift, speed = 3
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LSHIFT:
                    self.speed = 3
                    if self.last_action == "Up":
                        self.command['motion'] = [self.speed, 0]
                    elif self.last_action == "Down":
                        self.command['motion'] = [-self.speed, 0]
                    elif self.last_action == "Left":
                        self.command['motion'] = [0, self.speed]
                    elif self.last_action == "Right":
                        self.command['motion'] = [0, -self.speed]  
                    elif self.last_action == "None":
                        self.command['motion'] = [0, 0]
                        
                # unpress shift, speed = 1
                if event.type == pygame.KEYUP and event.key == pygame.K_LSHIFT:
                    self.speed = 1
                    if self.last_action == "Up":
                        self.command['motion'] = [self.speed, 0]
                    elif self.last_action == "Down":
                        self.command['motion'] = [-self.speed, 0]
                    elif self.last_action == "Left":
                        self.command['motion'] = [0, self.speed]
                    elif self.last_action == "Right":
                        self.command['motion'] = [0, -self.speed]  
                    elif self.last_action == "None":
                        self.command['motion'] = [0, 0]

                # remember the last action we took so we can perform at 1 speed or 3 speed
                # drive forward
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    self.command['motion'] = [self.speed, 0]
                    self.last_action = "Up"
                # drive backward
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    self.command['motion'] = [-self.speed, 0]
                    self.last_action = "Down"
                # turn left
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self.command['motion'] = [0, self.speed]
                    self.last_action = "Left"
                # drive right
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    self.command['motion'] = [0, -self.speed]    
                    self.last_action = "Right"
                # Stop
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.command['motion'] = [0, 0]
                    self.last_action = "None"
            ######################################################################################
            ######################################################################################

             # run SLAM
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.notification = 'SLAM is running'
                self.command['inference'] = True
                self.ekf_on = True
            # Cancel Auto Search
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.wp_manager.wps = []
                self.command['motion'] = [0, 0]
                self.b_auto_fruit_search = False
            # Save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # Make prediction of fruit position
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['make_prediction'] = True
            # Compare with truth map (if available)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                self.command['compare_truth'] = True
            # Run auto fruit search
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                if not self.b_auto_fruit_search:
                    self.ekf_on = True
                    self.command['inference'] = True
                    self.b_auto_fruit_search = True
                    self.init_fruit_search()
                    self.notification = 'Shopping'
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.notification = 'SLAM Map is cleared'
                    self.ekf.reset()
                    self.shopping_manager.reset()
                    self.wp_manager.wps = []
                    self.b_auto_fruit_search = False
                    self.ekf_on = False
            # Cycle Color mode
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                self.colors.next_mode()
            # Handle mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.command['click_map'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--yolo_model", default='yolov8_model')
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

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2
 
    operate = Operate(args,canvas)
    start = True
    save_time = time.time()
    while start:    
        

        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control_robot()
        operate.perform_auto_fruit_search()
        operate.detect_target()
        ekf_average_change = operate.update_slam(drive_meas)
        operate.record_data()
        #operate.compare_truth()

        loop_time = time.time() - save_time
        operate.draw(canvas)
        save_time = time.time()
        
        pygame.display.update()

        