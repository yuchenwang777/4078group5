# teleoperate the robot, perform SLAM and object detection

import os
import sys
import time
import cv2
import numpy as np

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations

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
        self.obstacles_hazard = float(args.hazard_radius)           #  (centre of aruco to camera needs to be > 17cm i think)
        self.a_star_res = float(args.path_res)                     #  grid resolution for astar
        self.hazard_boarder = float(args.hazard_boarder)           # distance to keep from edge of arena (hazard_boarder = 2 = 2 * res cm)
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


    # wheel control
    def control(self):
        if self.online:
            lv, rv = self.pibot.set_velocity(self.command['motion'])
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(lv, -rv, dt)
            self.control_clock = time.time()
        else:
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(0, 0, dt)
            self.control_clock = time.time()
        return drive_meas 

    # camera control
    def take_pic(self):
        if self.online:
            self.img = self.pibot.get_image()
        else:
            self.img = get_image(f'offline_img{self.offline_img_no}')

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):

        lms, self.aruco_img, rvecs, tvecs, ids = self.aruco_det.detect_marker_positions(self.img)

        if not self.ekf_on:
            return None
        
        lms = self.ekf.remove_unknown_lms(lms)
        lms = self.ekf.get_centre_pos(lms, rvecs, tvecs, ids)
        lms = self.ekf.change_landmark_covar(lms)

        if len(lms) > 0:
            self.b_saw_law_last_tick = True
        else:
            self.b_saw_law_last_tick = False

        # SLAM Algorithm
        self.ekf.predict(drive_meas)
        self.ekf.add_landmarks(lms)
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
            if self.detector_output:
                    for detection in self.detector_output:
                        b_found_new_fruit, valid_detection, b_update_wp_sample = self.shopping_manager.consider_detected_fruit(detection, self.ekf.robot.state)


    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.output.write_fruit_preds(self.shopping_manager.fruit_pose_dict)
            self.output.make_final_map()
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['make_prediction']:
            if self.detector_output:
                for detection in self.detector_output:
                    b_found_new_fruit, valid_detection, b_update_wp_sample = self.shopping_manager.consider_detected_fruit(detection, self.ekf.robot.state)
                    if valid_detection:
                        self.output.write_fruit_preds(self.shopping_manager.fruit_pose_dict)
                        self.notification = f'Fruit Preds updated'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['make_prediction'] = False
 
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

            self.command['motion'] = self.wp_manager.tick(self.ekf.robot, self.path_found, self.b_saw_law_last_tick, self.ekf.markers)

            if self.wp_manager.completed_route and self.shopping_manager.is_not_finished():
                self.shopping_manager.next_fruit()
                if self.shopping_manager.is_not_finished():
                    self.wp_manager.finished_waiting = False
                    print("UPDATE go next fruit")
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
                self.path_found = False
            else:
                self.path_found = True

            self.wp_manager.reset(new_wps, end_pos)
            self.wp_changed_last_iter = True
    
        # Init EKF
    
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
        fruits_list, fruits_true_pos, aruco_true_pos = read_true_map('lab_output/map_full.txt')
        lms = []
        for i in range(len(aruco_true_pos)):
            lmxy = [[aruco_true_pos[i][0]], [aruco_true_pos[i][1]]]
            lm = measure.Marker(position=lmxy, tag=i+1, covariance= (0.0*np.eye(2)))
            lms.append(lm)
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
        mouse_x, mouse_y = mouse_x - ekf_view_x, mouse_y + ekf_view_y - (2*self.v_pad)
        real_x, real_y = EKF.to_xy_coor((mouse_x, mouse_y),(ekf_view_w,ekf_view_h), self.ekf.m2pixel)
        robot_xy = self.ekf.robot.state[:2, 0].reshape((2, 1))
        pos_x, pos_y = robot_xy[0][0]+real_x, robot_xy[1][0]+real_y
        self.notification = f'x: {np.round(pos_x, 2)}, y: {np.round(pos_y, 2)}'
        self.command['click_map'] = False
        return None

    # paint the GUI            
    def draw(self, loop_time, ekf_average_change):

        bg_surface = pygame.Surface((self.gui_width,self.gui_height+100))
        bg_surface.fill(self.colors.COLOR_BG) # clear surface
        self.canvas.blit(bg_surface, (0, 0))

      

        # drawing slam map
        
        
        ekf_view = self.ekf.draw_slam_state(res=(self.ekf_view_width, self.ekf_view_height), # prev 320, 240
                                            wp_manager = self.wp_manager,
                                            grocery_idx = self.shopping_manager.grocery_idx,
                                            shopping_list = self.shopping_manager.shopping_list,
                                            fruit_pose_dict = self.shopping_manager.fruit_pose_dict,
                                            colors=self.colors)

        ekf_border_width = 20
        ekf_border_surface = pygame.Surface((self.ekf_view_width+ekf_border_width, self.ekf_view_height+ekf_border_width))
        ekf_border_surface.fill(self.colors.COLOR_EKF_VIEW_BORDER) # clear surface
        self.canvas.blit(ekf_border_surface, (2 * self.h_pad + 320 - ekf_border_width//2, self.v_pad - ekf_border_width//2))

        self.canvas.blit(ekf_view, (2 * self.h_pad + 320, self.v_pad)) # draw ekf on canvas
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(self.canvas, robot_view,
                                position=(self.h_pad, self.v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(self.canvas, detector_view,
                                position=(self.h_pad, 240 + 2 * self.v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(self.canvas, caption='SLAM', position=(2 * self.h_pad + 320, self.v_pad - 10))
        self.put_caption(self.canvas, caption='Detector',
                         position=(self.h_pad, 240 + 2 * self.v_pad))
        self.put_caption(self.canvas, caption='PiBot Cam', position=(self.h_pad, self.v_pad))
        self.put_caption(self.canvas, caption=f'{self.path_alg} Path', position=(320 + 3*self.h_pad + self.ekf_view_width, self.v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, self.colors.COLOR_HEADING_TEXT)       

        # printing text 
        # width, height = 1400, 660 for OVERALL CANVAS
        # A star and Text segments start at 1030 (x position)
        
        
        # robot status
        text_v_pad = 10
        text_h_pad = 10
        text_pad = 20
        font_size = 20


        font = pygame.font.SysFont(None, font_size)
        blank_surface = pygame.Surface((self.command_w,self.command_h))
        blank_surface.fill(self.colors.COLOR_COMMAND_BG) # clear surface
        
        font_status = pygame.font.SysFont(None, 30)
        status_text = self.wp_manager.state
        status_surface = font_status.render(status_text, True, self.colors.COLOR_TEXT) # robot status
        blank_surface.blit(status_surface, (text_h_pad, text_v_pad))  # robot coordinates

        # time since last seen LM
        last_seen_lm_text = f"Last seen LM: {float(time.time() - self.wp_manager.seen_last_landmark_timer):.1f} s"
        goal_surface = font.render(last_seen_lm_text, True, self.colors.COLOR_TEXT) 
        blank_surface.blit(goal_surface, (text_h_pad, 2*text_pad+text_v_pad))

        # distance to goal
        if self.wp_manager.distance_to_wp:
            goal_text = f"distance to goal: {float(self.wp_manager.distance_to_goal[0]):.3f}"
        else:
            goal_text = "distance to goal: N/A"
        goal_surface = font.render(goal_text, True, self.colors.COLOR_TEXT) 
        blank_surface.blit(goal_surface, (text_h_pad, 3*text_pad+text_v_pad))

        # distance to wp
        if self.wp_manager.distance_to_wp:
            goal_text = f"distance error: {float(self.wp_manager.distance_to_wp[0]):.3f}"
        else:
            goal_text = "distance error: N/A"
        goal_surface = font.render(goal_text, True, self.colors.COLOR_TEXT) 
        blank_surface.blit(goal_surface, (text_h_pad, 4*text_pad+text_v_pad))

         # desired heading (error)
        if self.wp_manager.desired_heading:
            error_text = f"heading error: {float(self.wp_manager.desired_heading[0]):.3f}"
        else:
            error_text = "heading error: N/A"
        error_surface = font.render(error_text, True, self.colors.COLOR_TEXT)
        blank_surface.blit(error_surface, (text_h_pad, 5*text_pad+text_v_pad))


        # robot coords
        coord_text = f"robot  x: {self.ekf.robot.state[:2, 0][0]:.1f}  y: {self.ekf.robot.state[:2, 0][1]:.1f}"
        coord_surface = font.render(coord_text, True, self.colors.COLOR_TEXT)  
        blank_surface.blit(coord_surface, (text_h_pad, 6*text_pad+text_v_pad))  # robot coordinates

        # print current fruit we are searching for
        if self.shopping_manager.is_not_finished():
            fruit_x, fruit_y = self.shopping_manager.get_fruit_to_find_pos()
            fruit_text = f"fruit   x: {fruit_x:.3f}     y: {fruit_y:.3f}"
            fruit_surface = font.render(fruit_text, True, self.colors.COLOR_TEXT)  
            blank_surface.blit(fruit_surface, (text_h_pad, 7*text_pad+text_v_pad))  
        else:
            fruit_text = f"fruit   x: N/A     y: N/A"
            fruit_surface = font.render(fruit_text, True, self.colors.COLOR_TEXT)  
            blank_surface.blit(fruit_surface, (text_h_pad, 7*text_pad+text_v_pad))  
            
        fps_text = f"Loop time: {loop_time:.3f}"
        fps_surface = font.render(fps_text, True, self.colors.COLOR_TEXT)  
        blank_surface.blit(fps_surface, (text_h_pad, 8*text_pad+text_v_pad)) 
        
        if ekf_average_change:
            ekf_change_text = f"EKF: {ekf_average_change:.5f}"
            ekf_change_surface = font.render(ekf_change_text, True, self.colors.COLOR_TEXT)  
            blank_surface.blit(ekf_change_surface, (text_h_pad, 9*text_pad+text_v_pad)) 
        else:
            ekf_change_text = f"EKF: N/A"
            ekf_change_surface = font.render(ekf_change_text, True, self.colors.COLOR_TEXT)  
            blank_surface.blit(ekf_change_surface, (text_h_pad, 9*text_pad+text_v_pad)) 

        canvas.blit(blank_surface, (1030, 430)) # add our text surface to our main canvas

        # Check list
        font = pygame.font.SysFont(None, 30)
        check_list_surface = pygame.Surface((self.command_w,self.command_h))
        check_list_surface.fill(self.colors.COLOR_COMMAND_BG) # clear surface
        
        canvas.blit(check_list_surface, (1030 + self.command_w + self.h_pad//2, 430))
        check_box_path = 'pics/check_small.png'
        not_check_box_path = 'pics/not_check_small.png'
        check_box_image = pygame.image.load(check_box_path) 
        check_box_image = pygame.transform.scale(check_box_image, (20, 20))
        not_check_box_image = pygame.image.load(not_check_box_path) 
        not_check_box_image = pygame.transform.scale(not_check_box_image, (20, 20))

        
        for i in range(len(self.shopping_manager.shopping_list)):
            fruit_text = self.shopping_manager.shopping_list[i][0]
            fruit_text_surface = font.render(fruit_text, True, self.colors.COLOR_TEXT) 
            check_list_surface.blit(fruit_text_surface, (40, text_pad + i*text_pad))

        canvas.blit(check_list_surface, (1030 + self.command_w + self.h_pad//2, 430))

        for i in range(len(self.shopping_manager.shopping_list)):
            img = None
            if i < self.shopping_manager.grocery_idx:
                img = check_box_image
            else:
                img = not_check_box_image   
            self.canvas.blit(img, (10 + 1030 + self.command_w + self.h_pad//2, text_pad + i*text_pad + 430 )) 

        

        
        
        self.put_caption(canvas, caption='Command Centre', position=(1030, 420))

        
        # draw notification on canvas
        self.canvas.blit(notifiation, (self.h_pad + 40, 596)) # BEFORE

        # count down
        max_time = 30 * 60  
        elapsed_time = time.time() - self.start_time

        # Draw Path
            # draw Path plot
        current_dir = os.getcwd()
        plot_image_path = os.path.join(current_dir, 'pics', 'path_plot.png')
        if not os.path.exists(plot_image_path):
            return 
        
        a_star_image = pygame.image.load(plot_image_path) 
        a_star_image = pygame.transform.scale(a_star_image, (350, 350))
        self.canvas.blit(pygame.transform.flip(a_star_image, True, True), (1030, self.v_pad)) 
        self.wp_changed_last_iter = False


        if elapsed_time < max_time:
            minutes = int(elapsed_time // 60)  
            seconds = int(elapsed_time % 60)  

            time_elapsed = f'{minutes}:{seconds:02d}'
            count_up_surface = TEXT_FONT.render(time_elapsed, False, self.colors.COLOR_TEXT)
        else:
            time_elapsed = "30:00 !!"
            count_up_surface = TEXT_FONT.render(time_elapsed, False, (255, 0, 0))

        # Blit the surface to the canvas
        self.canvas.blit(count_up_surface, (2*self.h_pad + 320 + self.ekf_view_width - 75, self.v_pad + 5))

        ekf_view_pos = (2 * self.h_pad + 320, self.v_pad)
        ekf_view_size = (self.ekf_view_width, self.ekf_view_height)
        self.output_coordinate_click(ekf_view_pos, ekf_view_size)

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
    parser.add_argument("--yolo_model", default='larry_large')
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

    width, height = 1400, 680 # prev: 700, 660
    #canvas = pygame.display.set_mode((width, height))
    canvas = None
 
    operate = Operate(args,canvas)
    start = True
    save_time = time.time()
    while start:    
        

        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.auto_fruit_search()
        operate.detect_target()
        ekf_average_change = operate.update_slam(drive_meas)
        operate.record_data()
        operate.compare_truth()

        loop_time = time.time() - save_time
        operate.draw(loop_time, ekf_average_change)
        save_time = time.time()
        
        pygame.display.update()

        