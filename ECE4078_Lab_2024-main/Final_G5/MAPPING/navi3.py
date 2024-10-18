# teleoperate the robot, perform SLAM and object detection
import re
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json
# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
from util.operate_util import *
from util.mapping_util import *
import pygame                       # python package for GUI
import shutil                       # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector_center as aruco # used our own aruco_detector which excludes markers > 10, and attempts to get the center of the cube.

sys.path.insert(0, "{}/path_planning".format(os.getcwd()))

# RRTC code from practical
from path_planning.RRTC import RRTC
from path_planning.Obstacle import *

# import YOLO components 
sys.path.insert(0, "{}/YOLO".format(os.getcwd()))
from YOLO.detector import Detector

class Operate:
    def __init__(self, args):
        # Uses maps generated in SLAM run
        self.fruit_list,self.fruit_true_pos,self.aruco_true_pos = read_lab_output(f'lab_output/{args.slam}',f'lab_output/{args.targets}')
        # self.fruit_list,self.fruit_true_pos,self.aruco_true_pos = read_true_map('Arena3.txt')
        self.current_fruit = 0
        # Prints for debugging
        print("fruit_list")
        print(self.fruit_list)
        print("fruit pos")
        print(self.fruit_true_pos)
        print("aruco pos")
        print(self.aruco_true_pos)
        self.search_list = read_search_list(args.search_list)
        print(self.search_list)

        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip,known_aruco_pos=self.aruco_true_pos)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = True
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Level 2 Nav'
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 3000
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        self.completed = False
        self.queued_actions = []
        self.closest_dist = 1e3
        self.stop_time = time.time()
        self.stopping = False
        # Path planning 
        # self.step_size = 0.10 #meters
        # Save planned route as list of lists of points (robot will stop after each sublist is completed)
        self.checkpoints = []

    def estimate_pose(self,camera_matrix, obj_info, robot_pose):

    # read in camera matrix (from camera calibration results)
        focal_length = camera_matrix[0][0]

    # there are 8 possible types of fruits and vegs
    ######### Replace with your codes #########
    # TODO: measure actual sizes of targets [width, depth, height] and update the dictionary of true target dimensions
        target_dimensions_dict = {'pear': [77/1000,71/1000,106/1000], 'lemon': [76/1000,51/1000,50/1000], 
                              'lime': [75/1000,52/1000,51/1000], 'tomato': [68/1000,70/1000,58/1000], 
                              'capsicum': [75/1000,70/1000,79/1000], 'potato': [96/1000,65/1000,60/1000], 
                              'pumpkin': [85/1000,83/1000,54/1000], 'garlic': [63/1000,61/1000,70/1000]}
    #########

    # estimate target pose using bounding box and robot pose
        target_class = obj_info[0]     # get predicted target label of the box
        target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
        true_height = target_dimensions_dict[target_class][2]   # look up true height of by class label

    # compute pose of the target based on bounding box info, true object height, and robot's pose
        pixel_height = target_box[3]
        pixel_center = target_box[0]
        distance = true_height/pixel_height * focal_length  # estimated distance between the robot and the centre of the image plane based on height
    # training image size 320x240p

    # remove estimates from far away, remove if statement if you want to keep all estimates
    
        image_width = 320 # change this if your training image is in a different size (check details of pred_0.png taken by your robot)
        x_shift = image_width/2 - pixel_center              # x distance between bounding box centre and centreline in camera view
        theta = np.arctan(x_shift/focal_length)     # angle of object relative to the robot
        ang = theta + robot_pose[2]     # angle of object in the world frame
    
   # relative object location
        distance_obj = distance/np.cos(theta) # relative distance between robot and object
        #print(f'object dist {distance_obj}')

        x_relative = distance_obj * np.cos(theta) # relative x pose
        y_relative = distance_obj * np.sin(theta) # relative y pose
        relative_pose = {'x': x_relative, 'y': y_relative}
    #print(f'relative_pose: {relative_pose}')

    # location of object in the world frame using rotation matrix
        delta_x_world = x_relative * np.cos(robot_pose[2]) - y_relative * np.sin(robot_pose[2])
        delta_y_world = x_relative * np.sin(robot_pose[2]) + y_relative * np.cos(robot_pose[2])
    # add robot pose with delta target pose
        target_pose = {'y': (robot_pose[1]+delta_y_world)[0],
                      'x': (robot_pose[0]+delta_x_world)[0]}
            
        return target_pose


    #print(f'delta_x_world: {delta_x_world}, delta_y_world: {delta_y_world}')
    #print(f'target_pose: {target_pose}')

        

    def get_next_instruction(self):
        """
        Converts next checkpoint into instructions
        """    
        stop_time = 5
        if len(self.checkpoints) == 0: # all sets of routes are completed
            self.completed = True
            return False
        else:
            try:
                self.notification = f"Navigating to {self.search_list[len(self.search_list)-len(self.checkpoints)]}"
            except IndexError:
                pass
        if len(self.checkpoints[0]) == 0: # Completed set
            print("Cleaning up")
            self.checkpoints.pop(0)
            return self.get_next_instruction()
        if len(self.checkpoints[0]) == 1: # last action in set
            self.drive_to_point(self.checkpoints[0][0])
            # self.checkpoints[0].pop(0)
            self.checkpoints.pop(0)
            self.queued_actions.append([0,0,stop_time])
            return True
        else:
            self.drive_to_point(self.checkpoints[0][0])
            self.checkpoints[0].pop(0)
            return True

    def play_action(self):
        """
        From contents of self.queued_actions sets self.command['motion'] to achieve the actions
        """
        if len(self.queued_actions) == 0: # if self.queued_actions is empty
            if self.get_next_instruction():
                self.play_action()
        else:
            next_action = self.queued_actions[0] 
            action_type = len(next_action)

            if action_type == 1:    # Target heading
                theta = self.get_robot_state()[2]
                ang_err = ang_clamp(next_action[0] - theta)

                if abs(ang_err) <= 0.1:
                    self.queued_actions.pop(0)
                elif ang_err <= 0:
                    self.command['motion'] = [0,-1]
                else:
                    self.command['motion'] = [0,1]
            elif action_type == 2: # Target coord
                #trying to update path based on current obstacles
                time.sleep(0.1)
                self.take_pic()
                time.sleep(0.1)
                #adjusting map based on fruits obstacles
                yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

                self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)
                print(self.detector_output)
                detections = []
                for j in range(len(self.detector_output)):
                #detection = self.detector_output[0]
                    print(j)
                    obj_pose = self.estimate_pose(self.camera_matrix,self.detector_output[j],self.get_robot_state())
                    if obj_pose:
                        print(obj_pose)
                        x_pose = obj_pose['x']
                        y_pose = obj_pose['y']
                        detected_fruit = self.detector_output[j][0]
                        print(detected_fruit)
                        distance = np.sqrt(x_pose**2 + y_pose**2)
                        print(distance)
                        if distance > 0.5:
                            print(f'ignoring {detected_fruit}')
                            continue
                        

                        #if detected_fruit in self.search_list:
                        fruit_index = 1000
                        print(f'found: {detected_fruit}')
                        for i in range(len(self.fruit_list)):
                            #print(self.fruit_list[i])
                            #print(detected_fruit)
                            if self.fruit_list[i] == detected_fruit:
                                fruit_index = i
                                break
                        #print(fruit_index)
                        detected_pose = [x_pose,y_pose]
                        final = [detected_pose, fruit_index]
                        print(final)
                        detections.append(final)
                    print(detections)
                    time.sleep(3)
                    if self.update_map_and_recalculate_path(detections):
                        return

            # covert the colour back for display purpose
                #self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR) 

                dist_to_target = self.get_distance(next_action)
                print(f"{dist_to_target}")
                if dist_to_target < self.closest_dist and not dist_to_target < 0.01:
                    self.closest_dist = dist_to_target
                    self.command['motion'] = [1,0]
                else:
                    if dist_to_target > self.closest_dist:
                        print("Missed point")
                    self.command['motion'] = [0,0]
                    self.closest_dist = 1e3
                    self.queued_actions.pop(0)
            elif action_type == 3: # Stop time
                if self.stop_time == 5:
                    print('target reached)')
                if time.time() > self.stop_time and self.stopping is True: # Stop completed
                    # print(f"Reached {self.search_list[len(self.search_list)-len(self.checkpoints)-1]}")
                    self.queued_actions.pop(0)
                    self.stopping = False
                    #print('arrived')
                    #self.current_fruit += 1
                else:
                    self.command['motion'] = self.queued_actions[0][:2]
                    if self.stopping == False:
                        self.stop_time = time.time() + next_action[2]
                        self.stopping = True
            else:
                print(f"Unexpected Arguement {self.queued_actions[0]}")

    def get_distance(self,waypoint):
        '''
        Takes x as first element of waypoint,y as second
        
        '''
        robot_state = self.get_robot_state()
        x = waypoint[0] - robot_state[0]
        y = waypoint[1] - robot_state[1]
        return np.hypot(x,y)

    def drive_to_point(self,waypoint):
        """
        Takes the next checkpoint and adds the desired heading and desired (x,y) into self.queued_actions
        For example:
        self.queued_actions = [[0.12],[0.3,0.4]] means to rotate until theta is 0.12, drive until (x,y) = (0.3,0.4)
        """
        robot_pose = self.get_robot_state()

        x = waypoint[0] - robot_pose[0]
        y = waypoint[1] - robot_pose[1]


        theta = np.arctan2(y,x) - robot_pose[2]
        target_heading = np.arctan2(y,x)
        print("Current heading",robot_pose[2])
        print("targeted heading",target_heading)
        if float(theta[0]) == 0:
            pass
        else:
            self.queued_actions.append([target_heading])
        self.queued_actions.append([0,0,0.5])
        self.queued_actions.append([waypoint[0],waypoint[1]])

    # wheel control
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # running in sim
        if args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        # running on physical robot (right wheel reversed)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()

        if self.data is not None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:  # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            # need to convert the colour before passing to YOLO
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)

            # covert the colour back for display purpose
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)

            # self.command['inference'] = False     # uncomment this if you do not want to continuously predict
            self.file_output = (yolo_input_img, self.ekf)

            # self.notification = f'{len(self.detector_output)} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip,known_aruco_pos=None):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')

        self.scale = scale
        
        self.baseline = baseline
        print("scale",scale)
        print("baseline",baseline)
        self.camera_matrix = camera_matrix

        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot,known_aruco_pos=known_aruco_pos)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                # image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 30
        # paint SLAM outputs

        ekf_view = self.ekf.draw_slam_state(self.fruit_true_pos, self.fruit_list,checkpoints=self.checkpoints,shopping_list=self.search_list, res=(600,600),
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

    def get_robot_state(self):
        return self.ekf.robot.state


    def navigate_to_fruits(self):
        """
        Calculates sets of points the robot must go to in order to arrive at all fruits on shoppping list
        self.checkpoints stores list of list of points , self.checkpoints[0] is the list of points to go to first fruit on shopping list
                                                         self.checkpoints[1] is the list of points to go from first fruit to second fruit 
                                                        .....
        """
        initial_pos = [0,0]

        # Define Obstacles from aruco xy & fruits xy
        fruit_true_pos = self.fruit_true_pos.tolist()
        aruco_true_pos = self.aruco_true_pos.tolist()

        FullObstacleCoorList = []

        for aruco_ind in range(len(aruco_true_pos)):
            FullObstacleCoorList.append(aruco_true_pos[aruco_ind])
                
        for fruit_xy in fruit_true_pos:
            FullObstacleCoorList.append(fruit_xy)
        # Get coordinates of fruits on shopping list    
        waypointList = []
        for item in self.search_list:
            waypointList.append(fruit_true_pos[self.fruit_list.index(item)])

        safe_radiuses = [0.35,0.3,0.25,0.2,0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.1] #revert to normal if takes too long
        bounds = {}

        # Iterate through shopping list to plan route between each waypoint
        for ind,waypoint in enumerate(waypointList):
            print("navigating to",waypoint)
            tempObstacleCoorList = FullObstacleCoorList.copy()

            tempObstacleCoorList.remove(waypoint)
            for radius in safe_radiuses:                
                obstacleList = create_obstacle_list(tempObstacleCoorList,radius)
                # print(len(obstacleList))
                routes = []
                print(f"{self.search_list[ind]} radius {radius}")
                
                for _ in range(10):
                    if len(self.checkpoints) == 0:
                        start = initial_pos
                    else:
                        start = self.checkpoints[-1][-1]
                    rrtc = RRTC(start=start,
                        goal=waypoint,
                        obstacle_list=obstacleList,
                        width = 2.7, #mm
                        height= 2.7, 
                        expand_dis=0.2, 
                        path_resolution=0.01, 
                        max_points=1000)#max_dis=0.7,
                    route = rrtc.planning()
                    if route is not None:
                        routes.append(route[1:-1])
                if len(routes) != 0:
                    for _ in range(100):
                        if len(self.checkpoints) == 0:
                            start = initial_pos
                        else:
                            start = self.checkpoints[-1][-1]
                        rrtc = RRTC(start=start,
                            goal=waypoint,
                            obstacle_list=obstacleList,
                            width = 2.7, #mm
                            height= 2.7, 
                            expand_dis=0.2, 
                            path_resolution=0.01, 
                            max_points=1000)#max_dis=0.7,
                        route = rrtc.planning()
                        if route is not None:
                            routes.append(route[1:-1])
                    # print(len(routes))
                    shortestRoute = min(routes,key=length_of_path)
                    # print(shortestRoute)
                    self.checkpoints.append(shortestRoute)
                    # print(shortestRoute)
                    print(f"Found path to {self.search_list[ind]} with safety radius {radius}")
                    bounds[self.search_list[ind]] = radius
                    break
    
    #if fruit in map is bad recalculates path
    def renavigate_to_fruits(self,start_index=0):
        """
        Calculates sets of points the robot must go to in order to arrive at all fruits on shoppping list
        self.checkpoints stores list of list of points , self.checkpoints[0] is the list of points to go to first fruit on shopping list
                                                         self.checkpoints[1] is the list of points to go from first fruit to second fruit 
                                                        .....
        """
        initial_pos = self.get_robot_state()
        print(initial_pos)
        self.checkpoints = []
        print(fruit_true_pos)
        

        # Define Obstacles from aruco xy & fruits xy
        fruit_true_pos = self.fruit_true_pos.tolist()
        aruco_true_pos = self.aruco_true_pos.tolist()

        FullObstacleCoorList = []

        for aruco_ind in range(len(aruco_true_pos)):
            FullObstacleCoorList.append(aruco_true_pos[aruco_ind])
                
        for fruit_xy in fruit_true_pos:
            FullObstacleCoorList.append(fruit_xy)
        # Get coordinates of fruits on shopping list    
        waypointList = []
        for item in self.search_list:
            print(start_index)
            if self.search_list.index(item) < start_index:
                continue
            else:
                waypointList.append(fruit_true_pos[self.fruit_list.index(item)])

        safe_radiuses = [0.35,0.3,0.25,0.2,0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.1] #revert to normal if takes too long
        bounds = {}

        # Iterate through shopping list to plan route between each waypoint
        for ind,waypoint in enumerate(waypointList):
            print("navigating to",waypoint)
            tempObstacleCoorList = FullObstacleCoorList.copy()

            tempObstacleCoorList.remove(waypoint)
            for radius in safe_radiuses:                
                obstacleList = create_obstacle_list(tempObstacleCoorList,radius)
                # print(len(obstacleList))
                routes = []
                print(f"{self.search_list[ind]} radius {radius}")
                
                for _ in range(10):
                    if len(self.checkpoints) == 0:
                        start = initial_pos
                    else:
                        start = self.checkpoints[-1][-1]
                    rrtc = RRTC(start=start,
                        goal=waypoint,
                        obstacle_list=obstacleList,
                        width = 2.7, #mm
                        height= 2.7, 
                        expand_dis=0.2, 
                        path_resolution=0.01, 
                        max_points=1000)#max_dis=0.7,
                    route = rrtc.planning()
                    if route is not None:
                        routes.append(route[1:-1])
                if len(routes) != 0:
                    for _ in range(30):
                        if len(self.checkpoints) == 0:
                            start = initial_pos
                        else:
                            start = self.checkpoints[-1][-1]
                        rrtc = RRTC(start=start,
                            goal=waypoint,
                            obstacle_list=obstacleList,
                            width = 2.7, #mm
                            height= 2.7, 
                            expand_dis=0.2, 
                            path_resolution=0.01, 
                            max_points=1000)#max_dis=0.7,
                        route = rrtc.planning()
                        if route is not None:
                            routes.append(route[1:-1])
                    # print(len(routes))
                    shortestRoute = min(routes,key=length_of_path)
                    # print(shortestRoute)
                    self.checkpoints.append(shortestRoute)
                    # print(shortestRoute)
                    print(f"Found path to {self.search_list[ind]} with safety radius {radius}")
                    bounds[self.search_list[ind]] = radius
                    break
        print(self.checkpoints)

    def update_map_and_recalculate_path(self, detections):

        #fruit_index = []
        #detected_pose = []
        change = False
        for i in range(len(detections)):
            fruit_index = detections[i][1]
            detected_pose = detections[i][0]
                    # Ignore poses outside the range -1.5 < x < 1.5 and -1.5 < y < 1.5
            if not (-1.5 < detected_pose[0] < 1.5 and -1.5 < detected_pose[1] < 1.5):
                print(f"Ignoring pose {detected_pose} for fruit index {fruit_index} as it is out of range.")
                continue
            map_pose = self.fruit_true_pos[fruit_index]
        #print(self.fruit_true_pos)
        #print(map_pose)
        #print(detected_pose)
            distance = self.calculate_distance(detected_pose, map_pose)
            if distance > 0.15:  # 15 cm
                print("Detected pose is different from map pose by more than 15 cm. Updating map and recalculating path.")
            # Update the map with the detected pose
                self.fruit_true_pos[fruit_index] = detected_pose
                change = True
        if change:
            # Recalculate the path from the current target index
            self.renavigate_to_fruits(start_index=self.current_fruit)
            # Stop the robot from driving into the obstacle
            self.command['motion'] = [0, 0]
            self.closest_dist = 1e3
            self.queued_actions = []
            return True
        return False

    def calculate_distance(self,pose1, pose2):
        return ((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2) ** 0.5
            

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                            False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--slam", type=str, default='slam.txt') # SLAM map from M3 
    parser.add_argument("--targets", type=str, default='targets.txt') # target map from M3
    parser.add_argument("--search_list", type = str, default="shopping_list.txt")
    # parser.add_argument("--yolo_model", default='YOLO/model/weights/200_Epochs.pt')
    #parser.add_argument("--yolo_model", default='YOLO/model/weights/18_oct.pt')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700+280, 660+100+20 # modified GUI due to larger slam map
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

    operate = Operate(args)
    operate.navigate_to_fruits()
    operate.queued_actions.append([0,0,3])
    # operate.find_my_location()
    #operate.plot_path_and_obstacles(operate.checkpoints, operate.fruit_true_pos, operate.aruco_true_pos)
    
    with open("planned_route"+".txt",'w') as label:
        label.write(f"{operate.checkpoints}")
    
    #input("Waiting to start")
    while start and operate.completed is False:
        # john = time.time()
        #operate.update_keyboard()
        if not operate.quit:
            operate.play_action()
        else:
            operate.command['motion'] = [0,0]
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        # operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        # print(time.time()-john)
