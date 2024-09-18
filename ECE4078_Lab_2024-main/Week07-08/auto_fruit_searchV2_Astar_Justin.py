import sys, os
import numpy as np
import json
import argparse
import time
import math
import heapq
from obstacles import *
from operate import Operate

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# Import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure
import matplotlib.pyplot as plt

# A* search algorithm for fruit searching
class AStarNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = float('inf')  # Cost from start to this node
        self.h = 0  # Heuristic cost to goal
        self.f = float('inf')  # Total cost

    def __lt__(self, other):
        return self.f < other.f

class AStar:
    def __init__(self, start, goal, obstacles, map_size, step_size=0.05, goal_threshold=0.3):
        self.start = AStarNode(start[0], start[1])
        self.start.g = 0
        self.start.f = self.heuristic(self.start, AStarNode(goal[0], goal[1]))
        self.goal = AStarNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.map_size = map_size
        self.step_size = step_size
        self.goal_threshold = goal_threshold

    def heuristic(self, node, goal):
        return math.hypot(node.x - goal.x, node.y - goal.y)

    def is_collision_free(self, node1, node2):
        for obstacle in self.obstacles:
            if obstacle.is_in_collision_with_points([(node1.x, node1.y), (node2.x, node2.y)]):
                return False
        return True

    def get_neighbors(self, node):
        neighbors = []
        directions = [
            (self.step_size, 0), (0, self.step_size),
            (-self.step_size, 0), (0, -self.step_size),
            (self.step_size, self.step_size), (self.step_size, -self.step_size),
            (-self.step_size, self.step_size), (-self.step_size, -self.step_size)
        ]
        for direction in directions:
            new_x = node.x + direction[0]
            new_y = node.y + direction[1]
            if -self.map_size / 2 <= new_x <= self.map_size / 2 and -self.map_size / 2 <= new_y <= self.map_size / 2:
                neighbor = AStarNode(new_x, new_y, node)
                if self.is_collision_free(node, neighbor):
                    neighbors.append(neighbor)
        return neighbors

    def search(self):
        open_list = []
        closed_set = set()
        heapq.heappush(open_list, (self.start.f, self.start))

        while open_list:
            _, current_node = heapq.heappop(open_list)
            if (current_node.x, current_node.y) in closed_set:
                continue
            closed_set.add((current_node.x, current_node.y))

            if self.heuristic(current_node, self.goal) < self.goal_threshold:
                return self.reconstruct_path(current_node)

            neighbors = self.get_neighbors(current_node)
            for neighbor in neighbors:
                if (neighbor.x, neighbor.y) in closed_set:
                    continue

                tentative_g = current_node.g + self.step_size
                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, self.goal)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node
                    heapq.heappush(open_list, (neighbor.f, neighbor))

        return None

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]

# Smooth the path to avoid sharp turns
def smooth_path(path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
    new_path = np.array(path)
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(new_path) - 1):
            for j in range(2):  # x and y
                aux = new_path[i][j]
                new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                new_path[i][j] += weight_smooth * (new_path[i - 1][j] + new_path[i + 1][j] - 2 * new_path[i][j])
                change += abs(aux - new_path[i][j])
    return new_path

# Dynamic obstacle avoidance mechanism
def dynamic_obstacle_avoidance(current_position, goal, obstacles, threshold=0.2):
    for obstacle in obstacles:
        dist_to_obstacle = np.linalg.norm(np.array([obstacle.center[0], obstacle.center[1]]) - np.array(current_position[:2]))
        if dist_to_obstacle < threshold:
            # Modify the path to avoid the obstacle
            new_goal = find_alternate_goal(current_position, goal, obstacle)
            return new_goal
    return goal

def find_alternate_goal(current_position, goal, obstacle):
    # Example: Simple right-hand rule to bypass the obstacle
    direction = np.arctan2(goal[1] - current_position[1], goal[0] - current_position[0])
    bypass_angle = np.pi / 4  # 45 degrees
    new_goal = [
        obstacle.center[0] + (obstacle.radius + 0.1) * np.cos(direction + bypass_angle),
        obstacle.center[1] + (obstacle.radius + 0.1) * np.sin(direction + bypass_angle)
    ]
    return new_goal

# Generating obstacles with an increased buffer zone
def generate_circular_obstacles(coordinates, robot_diameter=0.16, obstacle_diameter=0.18, safety_margin=0.1):
    obstacles = []
    # Effective radius includes robot radius, obstacle radius, and safety margin
    effective_radius = (obstacle_diameter / 2) + (robot_diameter / 2) + safety_margin
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
    target_positions = []
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print(f"{n_fruit}) {fruit} at [{fruit_true_pos[i][0]}, {fruit_true_pos[i][1]}]")
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
    angle_difference = desired_angle - th
    while angle_difference > np.pi:
        angle_difference -= 2 * np.pi
    while angle_difference <= -np.pi:
        angle_difference += 2 * np.pi

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
    distance = np.hypot(delta_x, delta_y)
    wheel_vel = 50  # tick

    try:
        drive_time = distance / (wheel_vel * scale)
        if np.isnan(drive_time) or drive_time <= 0:
            raise ValueError("Invalid drive time calculated.")
    except Exception as e:
        drive_time = 1  # Set a default drive time

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
    return [state[0].item(), state[1].item(), state[2].item()]

# Main loop
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

    # Read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    targetPose = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    combined_positions = np.vstack((fruits_true_pos, aruco_true_pos))

    for i in range(len(aruco_true_pos)):
        x, y = aruco_true_pos[i]
        ekf.add_true_landmarks(i + 1, x, y)

    obstacles = generate_circular_obstacles(combined_positions, safety_margin=0.1)
    start = [0.0, 0.0, 0.0]
    map_size = 2.9  # Should change to about 2.6 as the robot cannot touch the line

    full_path = []
    goal_indices = []
    current_position = start

    max_attempts = 5  # Limit the number of attempts to find a path

    for goal in targetPose:
        attempts = 0
        success = False
        while attempts < max_attempts:
            # Dynamically adjust the goal to avoid obstacles
            dynamic_goal = dynamic_obstacle_avoidance(current_position, goal, obstacles)
            pathfinder = AStar(current_position[:2], dynamic_goal[:2], obstacles, map_size)
            path = pathfinder.search()

            if path:
                path = smooth_path(path)  # Smooth the path
                for next_node in path[1:]:
                    lv, rv, dt = rotate_to_point(next_node, current_position)
                    current_position = get_robot_pose(ekf, robot, lv, rv, dt)
                    lv, rv, dt = drive_to_point(next_node, current_position)
                    current_position = get_robot_pose(ekf, robot, lv, rv, dt)
                    full_path.append(current_position.copy())

                    if np.linalg.norm(np.array(current_position[:2]) - np.array(goal[:2])) < 0.3:
                        print(f"Reached goal at coordinates: {goal[:2]}")
                        time.sleep(5)
                        goal_indices.append(len(full_path))
                        success = True
                        break

                if success:
                    break
            else:
                print(f"No path found to goal {goal[:2]} on attempt {attempts + 1}.")
                attempts += 1

        if not success:
            print(f"Failed to reach goal {goal[:2]} after {max_attempts} attempts. Moving to the next goal.")

    if len(full_path) > 0:
        print("Full path found!")
    else:
        print("No full path found.")


    # Visualization
    fig, ax = plt.subplots()
    ax.set_xlim(-map_size / 2, map_size / 2)
    ax.set_ylim(-map_size / 2, map_size / 2)

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
