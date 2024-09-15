# M4 - Autonomous fruit searching

# Basic Python packages
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
import random

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# Import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

# A* Node and A* Algorithm
import heapq

class AStarNode:
    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost  # g(n): Cost from start to current node
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

class AStar:
    def __init__(self, start, goal, obstacles, map_size, grid_res=0.1):
        self.start = AStarNode(start[0], start[1])
        self.goal = AStarNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.map_size = map_size
        self.grid_res = grid_res
        self.open_list = []
        self.closed_list = set()

    def heuristic(self, node):
        """Heuristic function to estimate the distance from the current node to the goal."""
        return ((node.x - self.goal.x) ** 2 + (node.y - self.goal.y) ** 2) ** 0.5

    def get_neighbors(self, node):
        """Get all valid neighbors of the current node."""
        neighbors = []
        directions = [(-self.grid_res, 0), (self.grid_res, 0), (0, -self.grid_res), (0, self.grid_res),
                      (-self.grid_res, -self.grid_res), (-self.grid_res, self.grid_res),
                      (self.grid_res, -self.grid_res), (self.grid_res, self.grid_res)]
        
        for d in directions:
            neighbor_x = node.x + d[0]
            neighbor_y = node.y + d[1]
            new_node = AStarNode(neighbor_x, neighbor_y, node.cost + self.grid_res, node)
            
            # Check if the neighbor collides with obstacles
            if not self.is_collision(new_node):
                neighbors.append(new_node)

        return neighbors

    def is_collision(self, node):
        """Check if a node collides with any obstacles."""
        for obstacle in self.obstacles:
            if obstacle.is_in_collision_with_points([(node.x, node.y)]):
                return True
        return False

    def build_astar_path(self):
        """A* algorithm to build the optimal path."""
        start_node = self.start
        goal_node = self.goal
        heapq.heappush(self.open_list, (self.heuristic(start_node), start_node))

        while self.open_list:
            _, current_node = heapq.heappop(self.open_list)

            # If we reached the goal, build the path
            if ((current_node.x - goal_node.x) ** 2 + (current_node.y - goal_node.y) ** 2) ** 0.5 < self.grid_res:
                return self.get_path(current_node)

            self.closed_list.add((round(current_node.x, 2), round(current_node.y, 2)))

            # Get neighbors and explore them
            for neighbor in self.get_neighbors(current_node):
                if (round(neighbor.x, 2), round(neighbor.y, 2)) in self.closed_list:
                    continue

                f_cost = neighbor.cost + self.heuristic(neighbor)
                heapq.heappush(self.open_list, (f_cost, neighbor))

        return None  # No path found

    def get_path(self, node):
        """Reconstruct the path from the goal to the start."""
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]


# Helper Functions
def read_search_list():
    """Read the search order of the target fruits."""
    search_list = []
    with open('M4_prac_shopping_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list

def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' positions in the search order."""
    print("Search order:")
    n_fruit = 1
    target_positions = []
    for fruit in search_list:
        for i in range(len(fruit_list)):  # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print(f'{n_fruit}) {fruit} at [{np.round(fruit_true_pos[i][0], 1)}, {np.round(fruit_true_pos[i][1], 1)}]')
                target_positions.append(fruit_true_pos[i])
        n_fruit += 1

    return target_positions

def rotate_to_point(waypoint, robot_pose):
    """Rotate the robot to face a specific waypoint."""
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # Calculate the angle to turn
    xg, yg = waypoint
    x, y, th = robot_pose

    desired_angle = math.atan2(yg - y, xg - x)
    current_angle = th
    angle_difference = desired_angle - current_angle

    # Normalize the angle to be within the range [-pi, pi]
    while angle_difference > math.pi:
        angle_difference -= 2 * math.pi
    while angle_difference <= -math.pi:
        angle_difference += 2 * math.pi

    wheel_vel = 30  # Rotation speed
    turn_time = abs(baseline * angle_difference * 0.5 / (scale * wheel_vel))

    print(f"Turning for {turn_time:.2f} seconds to face the waypoint.")

    if angle_difference < 0:
        lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
    else:
        lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)

    robot_pose[2] = desired_angle  # Update robot orientation
    return lv, rv, turn_time

def drive_to_point(waypoint, robot_pose):
    """Drive the robot to a specific waypoint."""
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # Calculate the distance to the waypoint
    delta_x = waypoint[0] - robot_pose[0]
    delta_y = waypoint[1] - robot_pose[1]
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    wheel_vel = 50  # Driving speed
    distance_error = distance

    # Calculate drive time
    try:
        drive_time = distance / (wheel_vel * scale)
        if math.isnan(drive_time) or drive_time <= 0:
            raise ValueError("Invalid drive time calculated.")
    except Exception as e:
        print(f"Error calculating drive time: {e}")
        drive_time = 1  # Set a default drive time

    print(f"Driving for {drive_time:.2f} seconds to reach the waypoint.")

    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    return lv, rv, drive_time

def get_robot_pose(ekf, robot, lv, rv, dt):
    """Update the robot's pose using EKF based on velocity and time."""
    drive_meas = measure.Drive(lv, -rv, dt)
    img = ppi.get_image_physical()
    aruco_detector = aruco.aruco_detector(robot)
    measurement, _ = aruco_detector.detect_marker_positions(img)
    ekf.predict(drive_meas)
    if measurement:
        ekf.update(measurement)

    # Update the robot pose [x, y, theta]
    state = ekf.get_state_vector()
    robot_pose = [state[0].item(), state[1].item(), state[2].item()]
    print(f"Actual pose: {robot_pose}")

    return robot_pose

# Main loop with A* replacing RRT
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='map.txt')  # change to 'M4_true_map_part.txt' for lv2&3
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

    # Read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    targetPose = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    combined_positions = np.vstack((fruits_true_pos, aruco_true_pos))

    # Add ArUco markers to the EKF
    for i in range(len(aruco_true_pos)):
        x, y = aruco_true_pos[i]
        ekf.add_true_landmarks(i + 1, x, y)

    # Generate obstacles
    obstacles = generate_circular_obstacles(combined_positions)
    start = [0.0, 0.0, 0.0]  # Robot start position
    map_size = 2.8  # Adjust map size accordingly

    # Initialize path list
    full_path = []
    goal_indices = []

    # Sequentially navigate to each goal
    current_position = start.copy()
    for goal in targetPose:
        while True:
            # A* pathfinding with debugging
            print(f"Planning path to goal: {goal[:2]}")
            astar = AStar(current_position[:2], goal[:2], obstacles, map_size, grid_res=0.1)
            path = astar.build_astar_path()
            if path is None:
                print(f"No path found to goal {goal[:2]}, stopping search.")
                break  # Exit the loop if no path found

            print(f"Path found: {path}")
            next_node = path[1]  # Get the next node in the path
            expected_orientation = math.atan2(next_node[1] - current_position[1], next_node[0] - current_position[0])
            print(f"Expected pose: {next_node}, Orientation: {math.degrees(expected_orientation):.2f} degrees")

            # Rotate to face the next node
            lv, rv, dt = rotate_to_point(next_node, current_position)
            current_position = get_robot_pose(ekf, robot, lv, rv, dt)  # Update the robot's pose
            full_path.append(current_position.copy())

            # Drive to the next node
            lv, rv, dt = drive_to_point(next_node, current_position)
            current_position = get_robot_pose(ekf, robot, lv, rv, dt)  # Update the robot's pose
            full_path.append(current_position.copy())

            # If EKF uncertainty is too high, perform additional corrections
            if ekf.P[0, 0] > 0.1 or ekf.P[1, 1] > 0.1:
                print("High uncertainty detected, performing corrections.")
                for _ in range(5):
                    lv, rv = ppi.set_velocity([1, 0], tick=30, time=1)
                    current_position = get_robot_pose(ekf, robot, lv, rv, 1)
                    full_path.append(current_position.copy())

            # Check if goal is reached
            if np.linalg.norm(np.array(current_position[:2]) - np.array(goal[:2])) < 0.3:
                print(f"Reached goal at coordinates: {goal[:2]}")
                goal_indices.append(len(full_path))
                break

    if full_path:
        print("Full path found!")
    else:
        print("No full path found.")
