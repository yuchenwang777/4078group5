# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import math
import argparse
from obstacles import *

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

class RRTC:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

        def __eq__(self, other):
            bool_list = []
            bool_list.append(self.x == other.x)
            bool_list.append(self.y == other.y)
            bool_list.append(np.all(np.isclose(self.path_x, other.path_x)))
            bool_list.append(np.all(np.isclose(self.path_y, other.path_y)))
            bool_list.append(self.parent == other.parent)
            return np.all(bool_list)
        
    def __init__(self, start=np.zeros(2),
                 goal=np.array([120,90]),
                 obstacle_list=None,
                 width=3,
                 height=3,
                 expand_dis=0.1, 
                 path_resolution=0.001, 
                 max_points=200):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle_list: list of obstacle objects
        width, height: search area
        expand_dis: min distance between random node and closest node in rrt to it
        path_resolution: step size to considered when looking for node to expand
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.width = width
        self.height = height
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_nodes = max_points
        self.obstacle_list = obstacle_list
        self.start_node_list = [] # Tree from start
        self.end_node_list = [] # Tree from end

    def grow_tree(self, tree, node):
        added_new_node = False
        nrst_index = self.get_nearest_node_index(tree, node)
        nrst_node = tree[nrst_index]

        new_node = self.steer(nrst_node, node, self.expand_dis)
        
        if self.is_collision_free(new_node):
            tree.append(new_node)
            added_new_node = True
        
        return added_new_node
    
    def check_trees_distance(self):
        for i in self.start_node_list:
            for j in self.end_node_list:
                dist, _ = self.calc_distance_and_angle(i, j)
                if dist <= self.expand_dis:
                    return True
        return False
    
    def planning(self):
        self.start_node_list = [self.start]
        self.end_node_list = [self.end]
        while len(self.start_node_list) + len(self.end_node_list) <= self.max_nodes:
            rnd_node = self.get_random_node()
            self.grow_tree(self.start_node_list, rnd_node)
            
            if self.check_trees_distance():
                path = self.generate_final_course(len(self.start_node_list) - 1, len(self.end_node_list) - 1)
                return self.smooth_path(path)
                
            rnd_node2 = self.get_random_node()
            self.grow_tree(self.end_node_list, rnd_node2)

            #if self.check_trees_distance():
            #    path = self.generate_final_course(len(self.start_node_list) - 1, len(self.end_node_list) - 1)
            #    return self.smooth_path(path)

            self.start_node_list, self.end_node_list = self.end_node_list, self.start_node_list

        return None
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * cos_theta
            new_node.y += self.path_resolution * sin_theta
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node
    
    def is_collision_free(self, new_node):
        if new_node is None:
            return True
        
        points = np.vstack((new_node.path_x, new_node.path_y)).T
        for obs in self.obstacle_list:
            in_collision = obs.is_in_collision_with_points(points)
            if in_collision:
                return False
        
        return True
    
    def is_direct_path_collision_free(self, start, end):
        """
        Check if the direct path between start and end points is collision-free.
        """
        x1, y1 = start
        x2, y2 = end
        path_x = np.linspace(x1, x2, num=int(np.hypot(x2 - x1, y2 - y1) / self.path_resolution))
        path_y = np.linspace(y1, y2, num=int(np.hypot(y2 - y1, x2 - x1) / self.path_resolution))
        points = np.vstack((path_x, path_y)).T
        for obs in self.obstacle_list:
            if obs.is_in_collision_with_points(points):
                return False
        return True
    
    def generate_final_course(self, start_mid_point, end_mid_point):
        node = self.start_node_list[start_mid_point]
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        
        node = self.end_node_list[end_mid_point]
        path = path[::-1]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path
    
    def smooth_path(self, path):
        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i:
                if self.is_direct_path_collision_free(path[i], path[j]):
                    smoothed_path.append(path[j])
                    i = j
                    break
                j -= 1
        return smoothed_path
    
    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if np.random.random() > 0.5:  # Bias towards goal
            x = self.end.x + self.width * (np.random.random_sample() - 0.5)
            y = self.end.y + self.height * (np.random.random_sample() - 0.5)
        else:
            x = self.width * (np.random.random_sample() - 0.5)
            y = self.height * (np.random.random_sample() - 0.5)
        rnd = self.Node(x, y)
        return rnd
    
    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):        
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    
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
    with open('M4_prac_shopping_list.txt', 'r') as fd:
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
    target_positions = []
    indexs = []
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
                target_positions.append(-1*fruit_true_pos[i])
                indexs.append(i)
        n_fruit += 1

     # Draw circles around the target fruits on the map image
    return target_positions, indexs


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 30 # tick
    
    # turn towards the waypoint
    turn_time = 0.0 # replace with your calculation
    print("Turning for {:.2f} seconds".format(turn_time))
    ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    
    # after turning, drive straight to the waypoint
    drive_time = 0.0 # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################

    return robot_pose

def on_click(event):
    print("Spacebar pressed. Closing GUI.")
    root.quit()

def on_space(event):
    global robot_pose, map_image_copy
    # Get the dimensions of the image
    img_width, img_height = map_image.size
    
    start = np.array([0.0,0.0])

    obstacles = [(x, y) for x, y in aruco_true_pos] + [(x, y) for x, y in obstaclePoses]
    #print(obstacles)
    obstacles_circles = [Circle(x, y, 0.1) for x, y in obstacles]  # Corrected line

       # Compute the path to each target fruit sequentially
    goals = []
    for target in targetPose:
        goals.append([target[0], target[1]])

    all_paths = []

    for i in range(len(goals)):
        rrt = RRTC(start=start, goal=np.array([goals[i][0], goals[i][1]]), obstacle_list=obstacles_circles)
        path = rrt.planning()
        start = np.array([goals[i][0], goals[i][1]])
        #print(path)

        if path is not None:
            all_paths.append(path)

    map_image_copy = map_image.copy()
    draw_all_paths_on_map(all_paths)

def draw_all_paths_on_map(paths):
    global map_image_copy
    draw = ImageDraw.Draw(map_image_copy)
    img_width, img_height = map_image_copy.size

    for path in paths:
        for i in range(len(path) - 1):
            # Convert coordinates from the map's coordinate system to image pixel coordinates
            x1 = int((path[i][0] + 1.5) / 3 * img_width)
            y1 = int((1.5 - path[i][1]) / 3 * img_height)
            x2 = int((path[i + 1][0] + 1.5) / 3 * img_width)
            y2 = int((1.5 - path[i + 1][1]) / 3 * img_height)
            draw.line((x1, y1, x2, y2), fill='red', width=2)

    # Update the image in the label
    map_photo.paste(map_image_copy)

#drawing the 0.5m radius circles around the target fruits, that the robot needs to be within to pick them up
def draw_target_circles(image, targets):
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    target_numbers = [1, 2, 3, 4, 5]
    counter = 0
    
    for target in targets:
        # Convert target coordinates to pixel coordinates
        # based on the direction of your x/y axis my need to adjust x and y (+/-) to get the correct orientation
        x = int((target[0] + 1.5) / 3 * img_width)
        y = int((1.5 - target[1]) / 3 * img_height)
        
        # Draw a circle with a 0.5m radius around the target
        radius = int(0.5 / 3 * img_width)  # Convert 0.5m to pixel radius
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline='blue', width=1)  # Increase width for better visibility
        #numbering the targets on the map so viewer can see the order them must go in
        #make sure orientation arrow is pointing to the target in order to "collect" it
        draw.text((x, y), str(target_numbers[counter]), fill='white', anchor='mm')
        counter += 1

# Draw the robot on the map image to show its current position and orientation
def draw_robot(x,y,orientation):
    global map_image_copy
    # Draw a red circle on the image at the specified coordinates
    draw = ImageDraw.Draw(map_image_copy)
    radius = 10
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')

    # Draw an arrow to represent the robot's orientation
    orientation = robot_pose[2]
    # Calculate the end point of the arrow
    arrow_length = 20  # Length of the arrow
    end_x = x + arrow_length * math.cos(orientation)
    end_y = y - arrow_length * math.sin(orientation)  # Invert y-axis for drawing

    # Draw the arrow
    draw.line((x, y, end_x, end_y), fill='red', width=3)

    
    # Update the image in the label
    map_photo.paste(map_image_copy)

def process_waypoints(x, y):
    # Convert pixel coordinates to actual waypoints if needed
    # For now, just print them
    print(f"Waypoint coordinates: ({x}, {y})")
    # estimate the robot's pose
    robot_pose = get_robot_pose()
    print(f"Robot pose: {robot_pose}")

# creates the map image based on the map given in the .txt file
# helpful for visualizing the robot's position and the target fruits, and placing waypoints
# allows for quick testing on different maps
def create_map_image(fruits_true_pos, aruco_true_pos, width=500, height=500):
    # Create a blank image
    map_image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(map_image)
    aruco_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    
    # based on the direction of your x/y axis my need to adjust x and y (+/-) to get the correct orientation
    # Draw the fruits
    for pos in fruits_true_pos:
        x = int((1.5 - pos[0]) / 3 * width)
        y = int((pos[1] + 1.5) / 3 * height)

        radius = 0.05 * width/3 # Radius of the fruit
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='green')
    
    # Draw the ArUco markers
    side = 0.07 * width/3  # Side length of the square in pixels
    for i, pos in enumerate(aruco_true_pos):
        x = int((1.5 - pos[0]) / 3 * width)
        y = int((pos[1] + 1.5) / 3 * height)
        # Drawing black blocks for markers
        draw.rectangle((x - side // 2, y - side // 2, x + side // 2, y + side // 2), fill='black')
        # Numbering the blocks
        draw.text((x, y), str(aruco_numbers[i]), fill='white', anchor='mm')

    # Draw the robot at the initial position
    robot_x = int((1.5 - robot_pose[0]) / 3 * width)
    robot_y = int((robot_pose[1] + 1.5) / 3 * height)
    radius = 5  # Radius of the robot
    draw.ellipse((robot_x - radius, robot_y - radius, robot_x + radius, robot_y + radius), fill='red')
    
    return map_image

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='map.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    targetPose, indexs = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    obstaclePoses = []
    print(indexs)
    for i in range(0,9):
        if i not in indexs:
            obstaclePoses.append(fruits_true_pos[i])
    print(obstaclePoses)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # Create the main window
    root = tk.Tk()
    root.title("Map Click Waypoints")

    # Load the map image
    #map_image = Image.open("M4_prac_map_layout_cropped.png")  # Update with your map image path
    map_image = create_map_image(fruits_true_pos, aruco_true_pos)

    #map_image_copy = map_image.copy()
    draw_target_circles(map_image, targetPose)
    map_photo = ImageTk.PhotoImage(map_image)

    # Create a label to display the map image
    map_label = tk.Label(root, image=map_photo)
    map_label.pack()

    # Bind the click event to the on_click function
    map_label.bind("<KeyPress-Return>", on_click)

    # Bind the spacebar event to the on_space function
    root.bind("<KeyPress-space>", on_space)

    # Start the GUI event loop
    root.mainloop()

    print(f"Updated robot pose: [{robot_pose[0]}, {robot_pose[1]}, {(180/math.pi)*robot_pose[2]}]")

        # robot drives to the waypoint
       # waypoint = [x,y]
        #drive_to_point(waypoint,robot_pose)
        #print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        #ppi.set_velocity([0, 0])
        #uInput = input("Add a new waypoint? [Y/N]")
        #if uInput == 'N':
        #    break

