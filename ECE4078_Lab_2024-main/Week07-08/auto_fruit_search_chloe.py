# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
from PIL import Image, ImageTk
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

# A* Pathfinding Algorithm
class AStar:
    def __init__(self, start, goal, obstacles, map_size, grid_res=0.1):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = obstacles
        self.map_size = map_size
        self.grid_res = grid_res
        self.open_set = []
        self.closed_set = []
        self.came_from = {}
        self.g_score = {}
        self.f_score = {}

    def heuristic(self, node):
        return np.linalg.norm(np.array(node) - np.array(self.goal))

    def is_in_obstacle(self, node):
        for obstacle in self.obstacles:
            if obstacle.is_in_collision_with_points([node]):
                return True
        return False

    def neighbors(self, node):
        directions = [(self.grid_res, 0), (-self.grid_res, 0), (0, self.grid_res), (0, -self.grid_res)]
        neighbors = [(node[0] + d[0], node[1] + d[1]) for d in directions]
        return [n for n in neighbors if not self.is_in_obstacle(n) and abs(n[0]) <= self.map_size / 2 and abs(n[1]) <= self.map_size / 2]

    def build_astar_path(self):
        self.open_set.append(self.start)
        self.g_score[self.start] = 0
        self.f_score[self.start] = self.heuristic(self.start)

        while self.open_set:
            current = min(self.open_set, key=lambda n: self.f_score.get(n, float('inf')))
            if np.linalg.norm(np.array(current) - np.array(self.goal)) < self.grid_res:
                return self.reconstruct_path(current)

            self.open_set.remove(current)
            self.closed_set.append(current)

            for neighbor in self.neighbors(current):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = self.g_score.get(current, float('inf')) + np.linalg.norm(np.array(current) - np.array(neighbor))

                if neighbor not in self.open_set:
                    self.open_set.append(neighbor)

                if tentative_g_score >= self.g_score.get(neighbor, float('inf')):
                    continue

                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.f_score[neighbor] = self.g_score[neighbor] + self.heuristic(neighbor)

        return None  # No path found

    def reconstruct_path(self, current):
        total_path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            total_path.insert(0, current)
        return total_path


def generate_circular_obstacles(coordinates, robot_diameter=0.16, obstacle_diameter=0.18):
    obstacles = []
    effective_radius = obstacle_diameter / 2 + robot_diameter / 2
    for (x, y) in coordinates:
        obstacles.append(Circle(x, y, effective_radius))
    return obstacles


def read_true_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

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
    search_list = []
    with open('M4_prac_shopping_list.txt', 'r') as fd:
        fruits = fd.readlines()
        for fruit in fruits:
            search_list.append(fruit.strip())
    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    print("Search order:")
    n_fruit = 1
    target_positions = []
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print(f'{n_fruit}) {fruit} at [{np.round(fruit_true_pos[i][0], 1)}, {np.round(fruit_true_pos[i][1], 1)}]')
                target_positions.append(fruit_true_pos[i])
        n_fruit += 1
    return target_positions


def rotate_to_point(waypoint, robot_pose):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    xg, yg = waypoint
    x, y, th = robot_pose
    
    desired_angle = np.arctan2(yg - y, xg - x)
    current_angle = th
    angle_difference = desired_angle - current_angle
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
    except Exception as e:
        drive_time = 1

    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    return lv, rv, drive_time


def get_robot_pose(ekf, robot, lv, rv, dt):
    drive_meas = measure.Drive(lv, -rv, dt)
    img = ppi.get_image_physical()
    aruco_detector = aruco.aruco_detector(robot)
    measurement, _ = aruco_detector.detect_marker_positions(img)
    ekf.predict(drive_meas)
    ekf.update(measurement)

    state = ekf.get_state_vector()
    robot_pose = [state[0].item(), state[1].item(), state[2].item()]
    print(f"Actual pose: {robot_pose}")
    return robot_pose


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='map.txt')
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

    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    targetPose = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    combined_positions = np.vstack((fruits_true_pos, aruco_true_pos))

    for i in range(len(aruco_true_pos)):
        x, y = aruco_true_pos[i]
        ekf.add_true_landmarks(i + 1, x, y)

    obstacles = generate_circular_obstacles(combined_positions)

    start = [0.0, 0.0, 0.0]
    map_size = 2.8
    full_path = []
    goal_indices = []

    current_position = start
    for goal in targetPose:
        while True:
            print(f"Planning path to goal: {goal[:2]}")
            astar = AStar(current_position[:2], goal[:2], obstacles, map_size, grid_res=0.1)
            path = astar.build_astar_path()
            if path is None:
                print(f"No path found to goal {goal[:2]}, stopping search.")
                break

            print(f"Path found: {path}")
            next_node = path[1]
            expected_orientation = math.atan2(next_node[1] - current_position[1], next_node[0] - current_position[0])
            print(f"Expected pose: {next_node}, Orientation: {math.degrees(expected_orientation):.2f} degrees")

            lv, rv, dt = rotate_to_point(next_node, current_position)
            current_position = get_robot_pose(ekf, robot, lv, rv, dt)
            full_path.append(current_position.copy())

            lv, rv, dt = drive_to_point(next_node, current_position)
            current_position = get_robot_pose(ekf, robot, lv, rv, dt)
            full_path.append(current_position.copy())

            if ekf.P[0, 0] > 0.1 or ekf.P[1, 1] > 0.1:
                for _ in range(5):
                    lv, rv = ppi.set_velocity([1, 0], tick=30, time=1)
                    current_position = get_robot_pose(ekf, robot, lv, rv, 1)
                    full_path.append(current_position.copy())

            if np.linalg.norm(np.array(current_position[:2]) - np.array(goal[:2])) < 0.3:
                print(f"Reached goal at coordinates: {goal[:2]}")
                goal_indices.append(len(full_path))
                break

    if full_path:
        print("Full path found!")
    else:
        print("No full path found.")
