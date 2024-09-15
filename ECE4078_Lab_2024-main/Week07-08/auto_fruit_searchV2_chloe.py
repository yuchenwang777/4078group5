import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import math
import random
import matplotlib.pyplot as plt
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

# Path finding algorithm for fruit searching (RRT)
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

# Generate circular obstacles
def generate_circular_obstacles(coordinates , robot_diameter=0.16, obstacle_diameter=0.18):
    obstacles = []
    effective_radius = obstacle_diameter/2 + robot_diameter / 2
    for (x, y) in coordinates:
        obstacles.append(Circle(x, y, effective_radius))
    return obstacles

# Main Loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip, args.port)

    # Load calibration parameters
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

    # Read the map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    targetPose = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    combined_positions = np.vstack((fruits_true_pos, aruco_true_pos))

    # Add ArUco markers to the EKF
    for i, (x, y) in enumerate(aruco_true_pos):
        ekf.add_true_landmarks(i + 1, x, y)

    # Generate obstacles
    obstacles = generate_circular_obstacles(combined_positions)
    start = [0.0, 0.0, 0.0]  # Robot start position
    map_size = 2.8

    current_position = start

    while True:
        # Run the EKF to predict and update the robot's pose every second
        img = ppi.get_image_physical()
        aruco_detector = aruco.aruco_detector(robot)
        measurement, _ = aruco_detector.detect_marker_positions(img)

        if measurement:
            # EKF update if a marker is detected
            ekf.update(measurement)
            current_position = ekf.get_state_vector()[:3]  # Update current robot position
            print(f"Position updated with marker: {current_position}")

            # After updating the position, re-run the RRT to find a new path
            rrt = RRT(current_position[:2], goal[:2], obstacles, map_size)
            path = rrt.build_rrt()
            if path:
                next_node = path[1]
                print(f"New path found, heading to: {next_node}")
            else:
                print("No valid path found.")

        else:
            # No marker detected, rotate the robot to search for markers
            print("No marker detected, rotating to search for one...")
            lv, rv, dt = rotate_to_find_marker(current_position)

        time.sleep(1)  # Run every second

# Function to rotate the robot until a marker is found
def rotate_to_find_marker(robot_pose):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    wheel_vel = 30  # Rotation speed
    angle_step = np.pi / 6  # Rotate by 30 degrees
    current_angle = robot_pose[2]
    new_angle = current_angle + angle_step

    # Normalize angle to within [-pi, pi]
    if new_angle > np.pi:
        new_angle -= 2 * np.pi
    if new_angle < -np.pi:
        new_angle += 2 * np.pi

    turn_time = abs(baseline * angle_step * 0.5 / (scale * wheel_vel))
    lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    robot_pose[2] = new_angle  # Update robot orientation
    return lv, rv, turn_time

# EKF Pose Update
def get_robot_pose(ekf, lv, rv, dt):
    drive_meas = measure.Drive(lv, -rv, dt)
    ekf.predict(drive_meas)

    state = ekf.get_state_vector()
    robot_pose = [state[0], state[1], state[2]]
    return robot_pose
