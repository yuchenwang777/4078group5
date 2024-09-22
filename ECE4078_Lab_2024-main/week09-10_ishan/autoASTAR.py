import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import math

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure      # measurements
import util.DatasetHandler as dh    # save/load functions
import pygame                       # python package for GUI
import shutil                       # python package for file operations

# ASTAR packages
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import matplotlib.pyplot as plt

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
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
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 300
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
            #self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
    def control(self, turn_tick=20):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'], turning_tick=turn_tick)
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
        self.take_pic()
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        self.ekf.predict(drive_meas)
        self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)
            self.file_output = (yolo_input_img, self.ekf)
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
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        self.scale=scale
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        self.baseline=baseline
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

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

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                            False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))
    def drive_to_point(self, waypoint_dict):
        ####################################################
        # TODO: replace with your codes to make the robot drive to the waypoint
        # One simple strategy is to first turn on the spot facing the waypoint,
        # then drive straight to the way point
        x_goal,y_goal = [waypoint_dict['x'],waypoint_dict['y']]
        x,y,th = self.get_robot_pose()
        print("Initial Pose:",x,y,th, self.baseline,self.scale)
        theta_final = np.arctan2(y_goal - y, x_goal - x)
        theta_diff = theta_final - th
        while theta_diff>np.pi:
            theta_diff-=np.pi*2
        while theta_diff<=-np.pi:
            theta_diff+=np.pi*2
        ln_dist = np.sqrt((x_goal - x) ** 2 + (y_goal- y) ** 2)
        wheel_vel = 50
        lin_time = ln_dist/(self.scale*wheel_vel) 
        wheel_vel=19 
        rot_time=abs(self.baseline*theta_diff*0.5/(self.scale*wheel_vel))
        print(f"Turning for {np.round(rot_time[0],2)} seconds")

        if theta_diff<0:
            self.command['motion'] = [0, -1]
        else:
            self.command['motion'] = [0, 1]
        
        operate.control_clock=time.time()
        rot_time += time.time()
        while time.time()<=rot_time:
            drive_meas = self.control()
            self.update_slam(drive_meas)
        
        print(f"Driving for {np.round(lin_time[0],2)} seconds")
        self.command['motion'] = [1, 0]
        lin_time += time.time()
        while time.time()<=lin_time:
            drive_meas = self.control()
            self.update_slam(drive_meas)
            if self.close_enough():
                return True
        print("Arrived at [{}, {}]".format(x_goal, y_goal))
        print("Final pose:",self.get_robot_pose().T)

        return False
    
    def get_robot_pose(self):
        ####################################################
        # TODO: replace with your codes to estimate the pose of the robot
        # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
        states = self.ekf.get_state_vector()
        robot_pose = states[0:3]
        ####################################################
        return robot_pose 
    # Custom function to check if robot is close to target
    def close_enough(self):
        robot_pos = self.get_robot_pose()[0:2]
        goal_pos = self.target
        distance = np.sum(np.square(goal_pos-robot_pos))
        if distance <= 0.25:
            return True
        return False
    
    def lost(self,threshold):
        cov_x = self.ekf.P[0,0]
        cov_y = self.ekf.P[1,1]
        if cov_x > threshold or cov_y > threshold:
            tick_speed = 15
            print('Robot is lost')
            turn_time=abs(self.baseline*2*np.pi*0.5/(self.scale*tick_speed))
            self.command['motion'] = [0, 1]
            operate.control_clock=time.time()
            turn_time += time.time()
            while time.time()<=turn_time and (cov_x > threshold or cov_y > threshold):
                drive_meas = self.control(turn_tick=tick_speed)
                self.update_slam(drive_meas)
                cov_x = self.ekf.P[0,0]
                cov_y = self.ekf.P[1,1]

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])
        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits
    @return: search order of the target fruits
    """
    search_list = []
    with open('shopping_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order
    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

def request_new_waypoint(): 
    x,y = 0.0,0.0
    x = input("X coordinate: ")
    try:
        x = float(x)
    except ValueError:
        print("Must be a number. Try Again")
        x=0.0
    y = input("Y coordinate: ")
    try:
        y = float(y)
    except ValueError:
        print("Must be a number. Try Again")
        y=0.0
    return x,y

"""Integration of ASTAR below was mainly obtained through documentation online"""

def map_building(target_fruit, fruit_list, fruit_pos, aruco_pos):
    # Create variables to store the position of the target fruit and other fruit
    target_fruit_pos = []
    obstacle_fruits = []
    for index, fruit in enumerate(fruit_list):
        if fruit in target_fruit:
            target_fruit_pos = fruit_pos[index]
            obstacle_fruits.append(fruit_pos[index])
        else:
            obstacle_fruits.append(fruit_pos[index])
    obstacle_fruits = np.array(obstacle_fruits)
    obstacles = np.append(obstacle_fruits, aruco_pos)
    obstacles = np.reshape(obstacles, (-1, 2))
    target_fruit_pos = np.array(target_fruit_pos)
    return target_fruit_pos, obstacles

def create_occupancy_grid(grid_precision, fruits_true_pos, aruco_true_pos):
    # Calculate the dimensions of the occupancy grid
    grid_precision = int(grid_precision) # 1 cm 
    robot_rad = int(np.floor(0.15 * grid_precision)) 
    fruit_rad = int(np.floor(0.08 * grid_precision)) + robot_rad 
    aruco_rad = int(np.floor(0.08 * grid_precision)) + robot_rad
    map_side_length = 3 
    grid_size = (map_side_length * grid_precision)
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=float)
    for marker in aruco_true_pos:
        centre_x = int(np.floor((marker[0] + 1.5) * grid_precision))
        centre_y = int(np.floor((marker[1] + 1.5) * grid_precision))
        for i in range(grid_size):
            for j in range(grid_size):
                if (i >= (centre_x - aruco_rad)) and (i <= (centre_x + aruco_rad)):
                    if (j >= (centre_y - aruco_rad)) and (j <= (centre_y + aruco_rad)):
                        occupancy_grid[i][j] = 1

    for i in range(grid_size):
        for j in range(grid_size):
            if (i <= robot_rad) or (i >= (3*grid_precision - robot_rad)):
                occupancy_grid[i][j] = 1
            if (j <= robot_rad) or (j >= (3*grid_precision - robot_rad)):
                occupancy_grid[i][j] = 1

    for fruit_pos in fruits_true_pos:
        for i in range(grid_size):
            for j in range(grid_size):
                dx = abs((fruit_pos[0]+ 1.5) * grid_precision - i)
                dy = abs((fruit_pos[1]+ 1.5) * grid_precision - j)

                if np.sqrt(dx**2 + dy**2) <= fruit_rad:
                    occupancy_grid[i][j] = 1

    return occupancy_grid

def on_line(point, line_start, line_end):
    x_point, y_point = point
    x1, y1 = line_start
    x2, y2 = line_end
    # Slope
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
    else:
        return x_point == x1
    # Y-int
    b = y1 - m * x1
    return y_point == m * x_point + b

def find_path_to_fruit(pos, search_list, search_index, fruits_list, fruits_true_pos, aruco_true_pos, grid_precision):
    for index, item in enumerate(fruits_list):
        if search_list[search_index] in item:
            target_index = index
    occupancy_grid = create_occupancy_grid(grid_precision, fruits_true_pos, aruco_true_pos)
    occupancy_grid = 1 - occupancy_grid
    save_occupancy_grid_as_image(occupancy_grid, "occupancy_grid.png")
    grid_obj = Grid(matrix = occupancy_grid)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    start_x, start_y = [pos[0][0],pos[1][0]]
    start_x = min(int(np.floor((start_x+1.5)*grid_precision)), (3* grid_precision - 1))
    start_y = min(int(np.floor((start_y+1.5)*grid_precision)), (3* grid_precision - 1))
    start = grid_obj.node(start_y, start_x)
    end_x = min(int(np.floor((fruits_true_pos[target_index][0]+1.5)*grid_precision)), (3* grid_precision - 1))
    end_y = min(int(np.floor((fruits_true_pos[target_index][1]+1.5)*grid_precision)), (3* grid_precision - 1))
    print('Fruit Centre at {0} , {1}'.format(end_x, end_y))
    closest_distance_target = float('inf')
    closest_distance_start = float('inf')
    closest_node = None
    for i in range(len(occupancy_grid)):
        for j in range(len(occupancy_grid)):
            if occupancy_grid[i][j] == 1:
                distance_to_target = math.dist([i, j], [end_x, end_y])
                distance_to_start = math.dist([i, j], [pos[0], pos[1]])
                if distance_to_target <= closest_distance_target:
                    closest_distance_target = distance_to_target
                    closest_node = grid_obj.node(j, i)
                    if distance_to_start <= closest_distance_start:
                        closest_distance_start = distance_to_start
                        closest_node = grid_obj.node(j, i)
    path, _ = finder.find_path(start, closest_node, grid_obj)
    min_node_dist = 0.4*grid_precision 
    index = len(path) - 2  
    while index > 0:
        if on_line([path[index].x, path[index].y], [path[index+1].x, path[index+1].y], [path[index-1].x, path[index-1].y]):
            if math.dist([path[index-1].x, path[index-1].y], [path[index+1].x, path[index+1].y]) < min_node_dist:
                del path[index]
        index -= 1
    
    return path, end_x, end_y
    
    
def save_occupancy_grid_as_image(occupancy_grid, image_filename):
    free_space_color = 255  # White
    obstacle_color = 0  # Black
    image = np.full_like(occupancy_grid, free_space_color, dtype=np.uint8)
    image[occupancy_grid == 1] = obstacle_color
    cv2.imwrite(image_filename, image)

# main loop
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    parser.add_argument("--map", type=str, default='lab_output/points.txt') # change to 'M4_true_map_part.txt' for lv2&3
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

    operate = Operate(args)

    #Import truth map, search list
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map(args.map)
    for i in range(len(aruco_true_pos)):
        x , y = aruco_true_pos[i]
        operate.ekf.add_fix_landmarks(i+1,x,y)
    search_list = read_search_list()

    print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)

    #AUTO FRUIT SEARCH
    waypoint = {"x":0.0,"y":0.0}
    while start:
        grid_precision = 50
        occupancy_grid = create_occupancy_grid(grid_precision, fruit_true_pos, aruco_true_pos)
        for search_index in range(len(search_list)-1):
            pos = operate.get_robot_pose()
            print("Current position:", pos)
            path, fruit_x, fruit_y = find_path_to_fruit(pos, search_list, search_index, fruit_list, fruit_true_pos, aruco_true_pos, grid_precision)
            operate.target = (fruit_x, fruit_y)
            for path_node in path:
                operate.lost(0.1)
                # x and y need to be flipped due to path finding quirk
                y = (path_node.x / grid_precision)-1.5
                x = (path_node.y / grid_precision)-1.5
                # Driving to waypoint
                waypoint['x']=x
                waypoint['y']=y
                if operate.drive_to_point(waypoint):
                    break
            
            print("Arrived at fruit:", search_list[search_index])
            operate.command['motion']= [0, 0]
            operate.control()
            time.sleep(2)
        print("oi ur done mate relax now")
        time.sleep(1234)
