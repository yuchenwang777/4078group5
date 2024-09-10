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
import random

OBSTACLE_SIZE = 0.07  # 30 cm in meters
ROBOT_RADIUS = 0.075  # 7.5 cm in meters

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def get_random_node(img_width, img_height):
    return Node(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))

def get_nearest_node(tree, random_node):
    nearest_node = tree[0]
    min_dist = distance(nearest_node, random_node)
    for node in tree:
        dist = distance(node, random_node)
        if dist < min_dist:
            nearest_node = node
            min_dist = dist
    return nearest_node

def is_collision(node, obstacles, obstacle_size=0.2, robot_radius=ROBOT_RADIUS):
    effective_size = obstacle_size + robot_radius
    for obs in obstacles:
        if abs(node.x - obs[0]) < effective_size / 2 and abs(node.y - obs[1]) < effective_size / 2:
            return True
    return False

def is_edge_collision(node1, node2, obstacles, step_size=0.001, obstacle_size=0.2, robot_radius=ROBOT_RADIUS):
    steps = int(distance(node1, node2) / step_size)
    for i in range(steps + 1):
        x = node1.x + i * (node2.x - node1.x) / steps
        y = node1.y + i * (node2.y - node1.y) / steps
        if is_collision(Node(x, y), obstacles, obstacle_size, robot_radius):
            return True
    return False

def extend_tree(tree, nearest_node, random_node, obstacles, step_size=0.1, robot_radius=ROBOT_RADIUS):
    theta = math.atan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
    new_node = Node(nearest_node.x + step_size * math.cos(theta), nearest_node.y + step_size * math.sin(theta))
    new_node.parent = nearest_node
    if not is_edge_collision(nearest_node, new_node, obstacles, step_size=0.001, robot_radius=robot_radius):
        tree.append(new_node)
        return new_node
    return None

def rrt(start, goal, img_width, img_height, obstacles, max_iter=1000, robot_radius=ROBOT_RADIUS):
    tree = [start]
    for _ in range(max_iter):
        random_node = get_random_node(img_width, img_height)
        nearest_node = get_nearest_node(tree, random_node)
        new_node = extend_tree(tree, nearest_node, random_node, obstacles)
        if new_node and distance(new_node, goal) < 0.1:
            goal.parent = new_node
            tree.append(goal)
            return tree
    return tree

def get_path(goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = node.parent
    return path[::-1]




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

def on_space(event):
    print("Spacebar pressed. Closing GUI.")
    root.quit()

def on_click(event):
    global robot_pose, map_image_copy
    img_width, img_height = map_image.size

    # Initialize the robot's starting position
    robot_pose = [0.0, 0.0, 0.0]
    start = Node(robot_pose[0], robot_pose[1])

    # Obstacles include Aruco markers and fruits not in the shopping list
    obstacles = [(x, y) for x, y in aruco_true_pos] + [(x, y) for x, y in obstaclePoses]
    
    # Define a list of colors to use for different segments of the path
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_index = 0

    # Compute the path to each target fruit sequentially
    full_path = []
    for target in targetPose:
        goal = Node(target[0], target[1])
        tree = rrt(start, goal, img_width, img_height, obstacles)
        path = get_path(goal)
        full_path.extend(path)
        start = goal  # Update the start to the last goal

    # Draw the full path as colored dots and lines connecting them
    map_image_copy = map_image.copy()
    draw = ImageDraw.Draw(map_image_copy)
    prev_node = None
    for node in full_path:
        x = int((node.x + 1.5) / 3 * img_width)
        y = int((1.5 - node.y) / 3 * img_height)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[color_index])
        if prev_node:
            prev_x = int((prev_node.x + 1.5) / 3 * img_width)
            prev_y = int((1.5 - prev_node.y) / 3 * img_height)
            draw.line((prev_x, prev_y, x, y), fill=colors[color_index], width=2)
        prev_node = node
        # Change color after each goal
        if (node.x, node.y) == (goal.x, goal.y):
            color_index = (color_index + 1) % len(colors)

    # Update the robot pose to the last node in the path
    if full_path:
        last_node = full_path[-1]
        theta = math.atan2(robot_pose[1] - last_node.y, robot_pose[0] - last_node.x)
        robot_pose = [last_node.x, last_node.y, theta]
        print(f"Updated robot pose: [{robot_pose[0]}, {robot_pose[1]}, {(180/math.pi)*robot_pose[2]}]")

    # Draw the robot at the final position
    draw_robot(int((robot_pose[0] + 1.5) / 3 * img_width), int((1.5 - robot_pose[1]) / 3 * img_height), robot_pose[2])

    draw = ImageDraw.Draw(map_image_copy)
    for node in path:
        x = int((node.x + 1.5) / 3 * img_width)
        y = int((1.5 - node.y) / 3 * img_height)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='red')

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
# Draw the robot on the map image to show its current position and orientation
def draw_robot(x, y, orientation):
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
def create_map_image(fruits_true_pos, aruco_true_pos, width=500, height=500, obstacle_size=OBSTACLE_SIZE):
    # Create a blank image
    map_image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(map_image)
    aruco_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Draw the fruits as circles with a diameter of 10 cm
    fruit_size = 0.1  # 10 cm in meters
    fruit_pixel_size = int(fruit_size / 3 * width)  # Convert 10 cm to pixel size

    for pos in fruits_true_pos:
        x = int((1.5 - pos[0]) / 3 * width)
        y = int((pos[1] + 1.5) / 3 * height)
        draw.ellipse((x - fruit_pixel_size // 2, y - fruit_pixel_size // 2, x + fruit_pixel_size // 2, y + fruit_pixel_size // 2), outline='green', width=2)
    
    # Draw the ArUco markers
    marker_size = 0.07  # Use the consistent obstacle size
    pixel_size = int(marker_size / 3 * width)  # Convert obstacle size to pixel size
    for i, pos in enumerate(aruco_true_pos):
        x = int((1.5 - pos[0]) / 3 * width)
        y = int((pos[1] + 1.5) / 3 * height)
        draw.rectangle((x - pixel_size // 2, y - pixel_size // 2, x + pixel_size // 2, y + pixel_size // 2), fill='black')
        draw.text((x, y), str(aruco_numbers[i]), fill='white', anchor='mm')

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
    map_label.bind("<Button-1>", on_click)

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

