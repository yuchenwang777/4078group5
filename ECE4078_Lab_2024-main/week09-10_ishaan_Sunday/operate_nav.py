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
                # try:
                #     object_distances = self.get_object_relative_pose()
                #     print(min(list(object_distances.values())))
                #     if min(list(object_distances.values())) < 0.08:
                #         print("Collision Detected!")
                #         print(object_distances,min(list(object_distances.values())))
                #         self.command['motion'] = [0,0]
                #         self.closest_dist = 1e3
                #         self.queued_actions.pop(0)
                # except ValueError:
                #     pass
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
                if time.time() > self.stop_time and self.stopping is True: # Stop completed
                    # print(f"Reached {self.search_list[len(self.search_list)-len(self.checkpoints)-1]}")
                    self.queued_actions.pop(0)
                    self.stopping = False
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

    def get_object_relative_pose(self):
        poses = {}
        for detection in self.detector_output:
            rel_pos = estimate_object_pose(self.camera_matrix,detection)
            dist = np.hypot(rel_pos['x'],rel_pos['y'])
            if detection[0] in poses.keys():
                if dist < poses[detection[0]]:
                    poses[detection[0]] = dist
                else:
                    pass
            else:
                poses[detection[0]] = dist
        return poses

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

        safe_radiuses = [0.35,0.3,0.25,0.2,0.15,0.1] # 
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
                    for _ in range(140):
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


    def navigate_to_next(self):
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

        safe_radiuses = [0.35,0.3,0.25,0.2,0.15,0.1] # 

        waypoint = fruit_true_pos[self.fruit_list[0]]
        # Iterate through shopping list to plan route between each waypoint
        print("navigating to",waypoint)

        FullObstacleCoorList.remove(waypoint)
        for radius in safe_radiuses:                
            obstacleList = create_obstacle_list(FullObstacleCoorList,radius)
            # print(len(obstacleList))
            routes = []
            print(f"{self.search_list[0]} radius {radius}")
            
            for _ in range(10):
                # print(f"{search_list[ind]} try {_} radius {radius}")
                start = self.get_robot_state()[:2]
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
                for _ in range(90):
                    # print(f"{search_list[ind]} try {_} radius {radius}")
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
                print(f"Found path to {self.search_list[0]} with safety radius {radius}")

                break
            

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


    # keyboard teleoperation, replace with your M1 codes if preferred        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0] + 1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0] - 1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1] + 1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1] - 1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            self.command['motion'] = [0, 0] 
            self.control()
            pygame.quit()
            sys.exit()

    def locate_update(self):

        self.take_pic()
        meas = self.control()

        self.update_slam(meas)


    def find_my_location(self):
        
        got_two = False

        rand = random.choice([-1, 1])

        start_time = time.time()

        while not got_two:

            self.locate_update()

            lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)

            i = 0

            condi = False
            for lm in lms:
                if lm.tag in self.ekf.taglist:
                    i+=1
                    if i >1:
                        condi = True

            time.sleep(0.2)

            if condi:
                self.pibot.set_velocity([0, 0])

                self.locate_update()

                got_two = True

            else:
                self.pibot.set_velocity([0, rand],time=0.2)

            passed_time = time.time() - start_time
            if passed_time >= 45:  
             break



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
    
    with open("planned_route"+".txt",'w') as label:
        label.write(f"{operate.checkpoints}")
    
    input("Waiting to start")
    while start and operate.completed is False:
        # john = time.time()
        operate.update_keyboard()
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
