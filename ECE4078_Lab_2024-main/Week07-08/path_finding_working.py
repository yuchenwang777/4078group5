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

import numpy as np
import random
import math
import matplotlib.pyplot as plt

class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class RRT:
    def __init__(self, start, goal, obstacles, map_size, step_size=0.1, goal_threshold=0.4, max_iter=1000):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.map_size = map_size
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iter = max_iter
        self.tree = [self.start]

    def get_random_point(self):
        return (random.uniform(-self.map_size/2, self.map_size/2), random.uniform(-self.map_size/2, self.map_size/2))

    def get_nearest_node(self, point):
        nearest_node = self.tree[0]
        min_dist = float('inf')
        for node in self.tree:
            dist = math.hypot(node.x - point[0], node.y - point[1])
            if dist < min_dist:
                nearest_node = node
                min_dist = dist
        return nearest_node

    def is_collision_free(self, node1, node2):
        for obstacle in self.obstacles:
            if obstacle.is_in_collision_with_points([(node1.x, node1.y), (node2.x, node2.y)]):
                return False
        return True

    def extend(self, nearest_node, random_point):
        theta = math.atan2(random_point[1] - nearest_node.y, random_point[0] - nearest_node.x)
        new_x = nearest_node.x + self.step_size * math.cos(theta)
        new_y = nearest_node.y + self.step_size * math.sin(theta)
        new_node = RRTNode(new_x, new_y, nearest_node)
        if self.is_collision_free(nearest_node, new_node):
            self.tree.append(new_node)
            return new_node
        return None

    def build_rrt(self):
        for _ in range(self.max_iter):
            random_point = self.get_random_point()
            nearest_node = self.get_nearest_node(random_point)
            new_node = self.extend(nearest_node, random_point)
            if new_node and math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < self.goal_threshold:
                return self.get_path(new_node)
        return None

    def get_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

#def main():
#    start = (0, 0)
#    goal = (1.5, 1.5)
#    map_size = 3.0
#    obstacles = [
#        Rectangle(np.array([0.5, 0.5]), 0.2, 0.2),
#        Circle(1.0, 1.0, 0.2)
#    ]

 #   rrt = RRT(start, goal, obstacles, map_size)
 #   path = rrt.build_rrt()

  #  if path:
   #     print("Path found!")
    #    for point in path:
     #       print(point)
    #else:
     #   print("No path found.")

    # Visualization
   # fig, ax = plt.subplots()
   # ax.set_xlim(-map_size/2, map_size/2)
   # ax.set_ylim(-map_size/2, map_size/2)

   # for obstacle in obstacles:
   #     if isinstance(obstacle, Rectangle):
   #         rect = plt.Rectangle(obstacle.origin, obstacle.width, obstacle.height, color='gray')
   #         ax.add_patch(rect)
   #     elif isinstance(obstacle, Circle):
   #         circle = plt.Circle(obstacle.center, obstacle.radius, color='gray')
   #         ax.add_patch(circle)

    #if path:
    #    path = np.array(path)
    #    ax.plot(path[:, 0], path[:, 1], '-o')

    #plt.show()
def generate_circular_obstacles(coordinates , robot_diameter=0.155, obstacle_diameter=0.1):
    """
    Generates a list of Circle obstacles from given coordinates and radii, adjusted for robot and obstacle diameters.

    :param coordinates: List of tuples representing the (x, y) coordinates of the circle centers.
    :param radii: List of radii for the circles.
    :param robot_diameter: Diameter of the robot (default is 0.155m).
    :param obstacle_diameter: Diameter of the obstacles (default is 0.1m).
    :return: List of Circle objects.
    """
    obstacles = []
    effective_radius = obstacle_diameter/2 + robot_diameter / 2
    for (x, y) in coordinates:
        obstacles.append(Circle(x, y, effective_radius))
    return obstacles


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
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
                target_positions.append(fruit_true_pos[i])
        n_fruit += 1

     # Draw circles around the target fruits on the map image
    return target_positions


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


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    targetPose = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    combined_positions = np.vstack((fruits_true_pos, aruco_true_pos))

    # Generate obstacles
    obstacles = generate_circular_obstacles(combined_positions)


    #waypoint = [0.0,0.0]
    #robot_pose = [0.0,0.0,0.0]
    start = (0.0, 0.0)
    map_size = 3.0 #should change to about 2.6 as robot cannot touch line

    # Initialize path list
    full_path = []

    # Sequentially navigate to each goal
    current_position = start
    for goal in targetPose:
        rrt = RRT(current_position, goal[:2], obstacles, map_size)
        path = rrt.build_rrt()
        if path:
            full_path.extend(path)
            current_position = path[-1]
        else:
            print(f"No path found to goal {goal[:2]}")
            break

    if full_path:
        print("Full path found!")
        for point in full_path:
            print(point)
    else:
        print("No full path found.")

# Visualization
fig, ax = plt.subplots()
ax.set_xlim(-map_size / 2, map_size / 2)
ax.set_ylim(-map_size / 2, map_size / 2)


# Draw obstacles
for obstacle in obstacles:
    if any(np.allclose(obstacle.center, target[:2], atol=0.1) for target in targetPose):
        circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, color='green')
        ax.add_patch(circle)
        # Draw 0.5m radius outline
        outline = plt.Circle((obstacle.center[0], obstacle.center[1]), 0.5, color='green', fill=False, linestyle='--')
        ax.add_patch(outline)
        # Add target number
        for j, target in enumerate(targetPose):
            if np.allclose(obstacle.center, target[:2], atol=0.1):
                ax.text(obstacle.center[0], obstacle.center[1], str(j + 1), color='white', ha='center', va='center')
                break
    elif any(np.allclose(obstacle.center, marker[:2], atol=0.1) for marker in aruco_true_pos):
        circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, color='black')
        ax.add_patch(circle)
        # Add ArUco marker number
        for i, marker in enumerate(aruco_true_pos):
            if np.allclose(obstacle.center, marker[:2], atol=0.1):
                ax.text(obstacle.center[0], obstacle.center[1], str(i), color='white', ha='center', va='center')
                break
    else:
        circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, color='gray')
        ax.add_patch(circle)

# Draw full path with different colors for each segment
if len(full_path) > 0:
    colors = ['red', 'blue', 'orange', 'purple', 'cyan']  # List of colors
    full_path = np.array(full_path)
    segment_start = 0
    for i, goal in enumerate(targetPose):
        # Calculate distances to the goal
        distances = np.linalg.norm(full_path - goal[:2], axis=1)
        segment_end = np.argmin(distances) + 1
        ax.plot(full_path[segment_start:segment_end, 0], full_path[segment_start:segment_end, 1], '-o', color=colors[i % len(colors)])
        segment_start = segment_end

plt.gca().invert_xaxis()  # Invert x-axis
plt.gca().invert_yaxis()  # Invert y-axis
plt.show()

    

