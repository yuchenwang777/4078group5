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
        # Setup directory to store dataset
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # Initialize the PenguinPi robot connection
        self.pibot = PenguinPi(args.ip, args.port)

        # Initialize SLAM parameters and ArUco marker detector
        self.ekf = self.init_ekf(args.calib_dir, args.ekf_threshhold)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)  # ArUco marker size: 7 cm

        # Output writer to save results
        self.output = dh.OutputWriter('lab_output')

        # Command dictionary to control robot behavior
        self.command = {
            'motion': [0, 0],           # robot's velocity commands
            'inference': False,          # object detection flag
            'output': False,             # flag to save SLAM map
            'make_prediction': False,    # flag to run inference
            'auto_fruit_search': False,  # flag for auto fruit search behavior
            'click_map': False,          # flag for map clicking feature
            'compare_truth': False       # flag for comparison with ground truth
        }

        # State management flags and initial values
        self.quit = False               # flag to stop the program
        self.file_output = None         # stores output data for predictions
        self.ekf_on = False             # SLAM state
        self.image_id = 0               # Image ID counter
        self.notification = 'Coordinates'  # Display notifications on GUI
        self.pred_notifier = False      # Prediction notifier for object detection
        
        # Timing and control variables
        self.start_time = time.time()   # Tracks overall operation time
        self.control_clock = time.time() # For wheel control timing

        # Initialize images for camera and ArUco marker detection
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)  # Main camera image
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)  # ArUco detection result

        # YOLO object detector initialization
        self.detector_output = None     # YOLO detection results
        self.detector = Detector(f'models/{args.yolo_model}.pt')  # YOLO model path
        self.yolo_vis = np.ones((240, 320, 3)) * 100  # YOLO visualization
        self.BACKGROUND_IMG = pygame.image.load('pics/gui_mask.jpg')  # GUI background

        # Robot speed and state
        self.speed = 1                  # Speed multiplier for robot
        self.last_action = "None"       # Last command action

        # Offline image ID for testing without a live camera feed
        self.offline_img_no = args.offline_id

        # GUI layout paddings and dimensions
        self.v_pad = 40
        self.h_pad = 20
        self.gui_width = 1000
        self.gui_height = 600
        self.ekf_view_width = 650
        self.ekf_view_height = 610
        self.caption_v_pad = 10
        self.command_w = 170
        self.command_h = 220

        # Canvas for GUI rendering
        self.canvas = canvas
        self.wp_changed_last_iter = False  # Indicates if waypoints were updated

        # Check if system is online or offline (for live testing vs simulation)
        self.online = args.online.upper() == "TRUE"

        # Waypoint and shopping manager
        self.wp_manager = wp_manager()   # Manages robot waypoints
        self.shopping_manager = shopping_manager()  # Manages shopping tasks in auto-search

        # Pathfinding algorithm selection based on user input
        if args.path_alg.upper()[0] == "A":
            self.path_alg = "A*"
        elif args.path_alg.upper()[0] == "D":
            self.path_alg = "Dijkstra"
        elif args.path_alg.upper()[0] == "T":
            self.path_alg = "A* Tom"

        # Various path and hazard configuration parameters
        self.obstacles_hazard = float(args.hazard_radius)  # Obstacle hazard radius
        self.a_star_res = float(args.path_res)             # A* resolution
        self.hazard_boarder = float(args.hazard_boarder)   # Hazard border radius

        # YOLO model and A* pathfinding timers
        self.a_star_delay = 500   # A* delay
        self.a_star_turning_delay = args.wp_random_update_interval  # Turning delay for A*
        self.a_star_timer = 0     # Timer for A* updates
        self.a_star_timer2 = time.time()  # Timer to control pathfinding behavior
        self.shopping_list_f = args.shopping_list  # Shopping list for auto fruit search
        self.b_saw_law_last_tick = False  # Tracks if last tick detected a law landmark
        self.colors = colors()  # GUI color management
        self.color_mode = 2  # Color mode (UI element)
        self.fruit_wp_threshhold = args.fruit_wp_threshhold  # Threshold for fruit waypoints

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
        new_wps, end_pos = path_alg(self, self.path_alg, fruit_to_find_pos, total_obstacles, self.colors)

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

        # Set the initial path state and update waypoints
        self.path_found = True
        self.update_waypoints()


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
        max_time = 2000
        elapsed_time = time.time() - self.start_time

        # Draw Path
        current_dir = os.getcwd()
        plot_image_path = os.path.join(current_dir, 'pics', 'path_plot.png')
        if not os.path.exists(plot_image_path):
            return 
        
        path_image = pygame.image.load(plot_image_path) 
        path_image = pygame.transform.scale(path_image, (350, 350))
        self.canvas.blit(pygame.transform.flip(path_image, True, True), (1030, self.v_pad)) 
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

        # PINGU!!!!!
        overlay_image = pygame.image.load('pingu.jpeg')
        overlay_image = pygame.transform.scale(overlay_image, (self.gui_width, self.gui_height + 100))
        self.canvas.blit(overlay_image, (0, 0))
        pygame.display.update()

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

    width, height = 1400, 680 # prev: 700, 660
    canvas = pygame.display.set_mode((width, height))
 
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
        operate.draw(loop_time, ekf_average_change)
        save_time = time.time()
        
        pygame.display.update()

        