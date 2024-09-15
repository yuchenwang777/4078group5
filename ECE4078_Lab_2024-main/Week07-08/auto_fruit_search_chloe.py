# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import tkinter as tk
from PIL import Image, ImageTk

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure


def get_obstacles(fruits_true_pos, aruco_true_pos):
    '''
    Gets list of obstacles across x and y axis
    '''
    ox, oy = [], []
    for fruit in fruits_true_pos:
        ox.append(fruit[0])
        oy.append(fruit[1])
    for marker in aruco_true_pos:
        ox.append(marker[0])
        oy.append(marker[1])
    
    return ox, oy


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits & vegs to search for"""
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
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print(f'{n_fruit}) {fruit} at [{np.round(fruit_true_pos[i][0], 1)}, {np.round(fruit_true_pos[i][1], 1)}]')
        n_fruit += 1


# Waypoint navigation
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    # Simple turn and move
    wheel_vel = 50 # tick
    
    # Calculate turn angle
    angle = np.arctan2((waypoint[1] - robot_pose[1]), (waypoint[0] - robot_pose[0])) - robot_pose[2]
    turn_time = abs(angle) * baseline / wheel_vel  # replace with your calculation
    print(f"Turning for {turn_time:.2f} seconds")
    ppi.set_velocity([0, 1 if angle > 0 else -1], turning_tick=wheel_vel, time=turn_time)
    
    # Calculate drive distance
    distance = np.sqrt((waypoint[0] - robot_pose[0]) ** 2 + (waypoint[1] - robot_pose[1]) ** 2)
    drive_time = distance / (scale * wheel_vel)  # replace with your calculation
    print(f"Driving for {drive_time:.2f} seconds")
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    print(f"Arrived at [{waypoint[0]}, {waypoint[1]}]")


def get_robot_pose():
    """Estimate the robot's pose using SLAM"""
    # Dummy SLAM code placeholder; replace with actual SLAM implementation
    robot_pose = [0.0, 0.0, 0.0]  # [x, y, theta]
    return robot_pose


# Main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt')  # Use 'M4_true_map_part.txt' for level 2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip, args.port)

    # Read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0, 0.0]
    robot_pose = [0.0, 0.0, 0.0]

    # Get obstacles (ox, oy)
    ox, oy = get_obstacles(fruits_true_pos, aruco_true_pos)

    # The following is only a skeleton code for semi-auto navigation
    while True:
        # Enter the waypoints
        x, y = input("X coordinate of the waypoint (cm): "), input("Y coordinate of the waypoint (cm): ")
        try:
            x, y = float(x) * 100.0, float(y) * 100.0
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
            continue

        # Estimate the robot's pose (SLAM)
        robot_pose = get_robot_pose()

        # Navigate to the waypoint
        waypoint = [x, y]
        drive_to_point(waypoint, robot_pose)
        print(f"Finished driving to waypoint: {waypoint}; New robot pose: {robot_pose}")

        # Exit or continue
        ppi.set_velocity([0, 0])
        if input("Add a new waypoint? [Y/N]") == 'N':
            break
