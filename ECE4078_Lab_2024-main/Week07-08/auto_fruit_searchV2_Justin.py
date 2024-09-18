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
from obstacles import *
from operate import Operate

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

import random
import matplotlib.pyplot as plt

# path finding algorithm for fruit searching
class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class RRT:
    def __init__(self, start, goal, obstacles, map_size, step_size=0.1, goal_threshold=0.3, max_iter=5000, goal_bias=0.5):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.map_size = map_size
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.tree = [self.start]

    def get_random_point(self):
        if random.random() < self.goal_bias:
            return (self.goal.x, self.goal.y)
        else:
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

# generating the obstacles taking into account size of robot and obstacle size
def generate_circular_obstacles(coordinates , robot_diameter=0.16, obstacle_diameter=0.18):
    obstacles = []
    effective_radius = obstacle_diameter/2 + robot_diameter / 2
    for (x, y) in coordinates:
        obstacles.append(Circle(x, y, effective_radius))
    return obstacles

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for"""
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
    """Read the search order of the target fruits"""
    search_list = []
    with open('M4_prac_shopping_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list

def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order"""
    print("Search order:")
    n_fruit = 1
    target_positions = []
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
                target_positions.append(fruit_true_pos[i])
        n_fruit += 1
    return target_positions

def rotate_to_point(waypoint, robot_pose):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # Calculate the angle to turn
    xg, yg = waypoint
    x, y, th = robot_pose

    desired_angle = np.arctan2(yg - y, xg - x)
    angle_difference = desired_angle - th
    while angle_difference > np.pi:
        angle_difference -= np.pi * 2
    while angle_difference <= -np.pi:
        angle_difference += np.pi * 2

    wheel_vel = 30  # tick
    turn_time = abs(baseline * angle_difference * 0.5 / (scale * wheel_vel))
    robot_pose[2] = desired_angle

    if angle_difference < 0:
        lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
    else:
        lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)

    return lv, rv, turn_time

def drive_to_point(waypoint, robot_pose):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    delta_x = waypoint[0] - robot_pose[0]
    delta_y = waypoint[1] - robot_pose[1]

    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    wheel_vel = 50  # tick

    try:
        drive_time = distance / (wheel_vel * scale)
        if np.isnan(drive_time) or drive_time <= 0:
            raise ValueError("Invalid drive time calculated.")
    except Exception as e:
        drive_time = 1  # Set a default drive time

    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    print(f"Arrived at [{waypoint[0]}, {waypoint[1]}]")
    return lv, rv, drive_time

def get_robot_pose(ekf, robot, lv, rv, dt):
    print("Starting EKF Prediction Step...")
    drive_meas = measure.Drive(lv, -rv, dt)
    print(f"Drive Measurements: lv={lv}, rv={rv}, dt={dt}")
    
    img = ppi.get_image_physical()
    aruco_detector = aruco.aruco_detector(robot)
    measurement, _ = aruco_detector.detect_marker_positions(img)
    
    print("Performing EKF Prediction...")
    ekf.predict(drive_meas)
    print("EKF Prediction Step Complete.")
    
    print("Performing EKF Update with Measurements...")
    if measurement:
        print(f"Measurements Detected: {measurement}")
    else:
        print("No measurements detected.")
        
    ekf.update(measurement)
    print("EKF Update Step Complete.")
    
    # update the robot pose [x,y,theta]
    state = ekf.get_state_vector()
    robot_pose = [state[0].item(), state[1].item(), state[2].item()]
    print(f"Updated Robot Pose: {robot_pose}")
    print("-----------------------------------------------------")

    return robot_pose

def rotate_to_face_goal(goal, robot_pose):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    xg, yg = goal
    x, y, th = robot_pose
    
    desired_angle = np.arctan2(yg - y, xg - x)
    angle_difference = desired_angle - th
    while angle_difference > np.pi:
        angle_difference -= np.pi * 2
    while angle_difference <= -np.pi:
        angle_difference += np.pi * 2

    wheel_vel = 30  # tick
    turn_time = abs(baseline * angle_difference * 0.5 / (scale * wheel_vel))
    robot_pose[2] = desired_angle

    if angle_difference < 0:
        lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
    else:
        lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    
    time.sleep(4)
    return lv, rv, turn_time

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='map.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip, args.port)

    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    fileK = "calibration/param/intrinsic.txt"
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')

    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    ekf = EKF(robot)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    targetPose = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    combined_positions = np.vstack((fruits_true_pos, aruco_true_pos))

    for i in range(len(aruco_true_pos)):
        x, y = aruco_true_pos[i]
        ekf.add_true_landmarks(i + 1, x, y)

    # Generate obstacles
    obstacles = generate_circular_obstacles(combined_positions)

    start = [0.0, 0.0, 0.0]
    map_size = 2.9  # should change to about 2.6 as robot cannot touch line

    # Initialize path list
    full_path = []
    goal_indices = []

    # Sequentially navigate to each goal
    current_position = start
    while True:
        for goal in targetPose:
            while True:
                rrt = RRT(current_position[:2], goal[:2], obstacles, map_size)
                path = rrt.build_rrt()
                if path:
                    for next_node in path[1:]:  # Go through each node in the path
                        # Rotate towards the next node
                        expected_orientation = np.arctan2(next_node[1] - current_position[1], next_node[0] - current_position[0])
                        print(f"expected pose: {next_node, expected_orientation}")
                        lv, rv, dt = rotate_to_point(next_node, current_position)
                        current_position = get_robot_pose(ekf, robot, lv, rv, dt)  # Update the robot's pose theta
                        
                        # Drive to the next node
                        lv, rv, dt = drive_to_point(next_node, current_position)
                        current_position = get_robot_pose(ekf, robot, lv, rv, dt)  # Update the robot's pose x and y

                        # Save the current position to the full path
                        full_path.append(current_position.copy())

                        # If the current position is close to the goal, break out of the loop
                        if np.linalg.norm(np.array(current_position[:2]) - np.array(goal[:2])) < 0.3:
                            print(f"Reached goal at coordinates: {goal[:2]}")
                            time.sleep(5)
                            goal_indices.append(len(full_path))
                            break

                    # After reaching the goal, move on to the next goal
                    break
                else:
                    # No path found, try again or move back to the start
                    print(f"No path found to goal {goal[:2]} going back to origin")
                    for node in reversed(full_path):
                        lv, rv, dt = rotate_to_point(node[:2], current_position)
                        current_position = get_robot_pose(ekf, robot, lv, rv, dt)  # Update the robot's pose theta
                        lv, rv, dt = drive_to_point(node[:2], current_position)
                        current_position = get_robot_pose(ekf, robot, lv, rv, dt)  # Update the robot's pose x and y
                        print(f"Tracing back to: {node[:2]}")
                    current_position = start
                    full_path = [start]

        if len(full_path) > 0:
            print("Full path found!")
        else:
            print("No full path found.")

        # Visualization code here...

        fig, ax = plt.subplots()
        ax.set_xlim(-map_size / 2, map_size / 2)
        ax.set_ylim(-map_size / 2, map_size / 2)

        # Draw obstacles
        for obstacle in obstacles:
            if any(np.allclose(obstacle.center, target[:2], atol=0.1) for target in targetPose):
                circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, color='green')
                ax.add_patch(circle)
                outline = plt.Circle((obstacle.center[0], obstacle.center[1]), 0.5, color='green', fill=False, linestyle='--')
                ax.add_patch(outline)
                for j, target in enumerate(targetPose):
                    if np.allclose(obstacle.center, target[:2], atol=0.1):
                        ax.text(obstacle.center[0], obstacle.center[1], str(j + 1), color='white', ha='center', va='center')
                        break
            elif any(np.allclose(obstacle.center, marker[:2], atol=0.1) for marker in aruco_true_pos):
                circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, color='black')
                ax.add_patch(circle)
                for i, marker in enumerate(aruco_true_pos):
                    if np.allclose(obstacle.center, marker[:2], atol=0.1):
                        ax.text(obstacle.center[0], obstacle.center[1], str(i+1), color='white', ha='center', va='center')
                        break
            else:
                circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, color='gray')
                ax.add_patch(circle)

        if len(full_path) > 0:
            colors = ['red', 'green', 'orange', 'purple', 'cyan']
            full_path = np.array(full_path)
            segment_start = 0
            for i, goal_index in enumerate(goal_indices):
                segment_end = goal_index
                ax.plot(full_path[segment_start:segment_end, 0], full_path[segment_start:segment_end, 1], '-o', color=colors[i % len(colors)])
                
                for j in range(segment_start, segment_end):
                    x, y, theta = full_path[j]
                    dx = 0.1 * np.cos(theta)
                    dy = 0.1 * np.sin(theta)
                    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='black')
                
                segment_start = segment_end

            if segment_start < len(full_path):
                ax.plot(full_path[segment_start:, 0], full_path[segment_start:, 1], '-o', color=colors[len(goal_indices) % len(colors)])
                
                for j in range(segment_start, len(full_path)):
                    x, y, theta = full_path[j]
                    dx = 0.1 * np.cos(theta)
                    dy = 0.1 * np.sin(theta)
                    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='black')

        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.show()