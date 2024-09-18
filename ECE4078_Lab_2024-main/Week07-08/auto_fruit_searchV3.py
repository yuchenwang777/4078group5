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

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# path finding algorithm for fruit searching
class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class RRT:
    def __init__(self, start, goal, obstacles, map_size, step_size=0.05, goal_threshold=0.3, max_iter=1000, goal_bias=0.3):
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
            if self.line_intersects_circle((node1.x, node1.y), (node2.x, node2.y), obstacle.center, obstacle.radius):
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
                path = self.get_path(new_node)
                return self.smooth_path(path)
        return None

    def get_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    
    def line_intersects_circle(self, p1, p2, center, radius):
        # Vector from p1 to p2
        d = np.array(p2) - np.array(p1)
        # Vector from p1 to center
        f = np.array(p1) - np.array(center)
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return False
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        if (t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1):
            return True
        return False

    def smooth_path(self, path):
        if len(path) < 3:
            return path

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i:
                if self.is_collision_free(RRTNode(path[i][0], path[i][1]), RRTNode(path[j][0], path[j][1])):
                    smoothed_path.append(path[j])
                    i = j
                    break
                j -= 1
        return smoothed_path

# genrating the obstacles taking into account size of robot and obstacle size
# just making markers cirlces for now as im lazy can change to rectangles later if we want
def generate_circular_obstacles(coordinates , robot_diameter=0.16, obstacle_diameter=0.24):
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
def rotate_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # Calculate the angle to turn
    xg, yg = waypoint
    x,y,th = robot_pose
    lv,rv,turn_time = 0,0,0
    
    
    # Normalize the angle to be within the range [-pi, pi]
    desired_angle = np.arctan2(yg - y, xg - x)
    current_angle = th
    angle_difference = desired_angle - current_angle
    while angle_difference>np.pi:
        angle_difference-=np.pi*2
    while angle_difference<=-np.pi:
        angle_difference+=np.pi*2

    wheel_vel = 30  # tick
    Kp = 0.5 # Proportional gain/ may need to change for better performance, if this works at all

    while abs(angle_difference) > 0.001:  # Continue rotating until the angle error is small about 3 degrees can decrease if needed
        # Calculate the time to turn
        turn_time = Kp*abs(baseline * angle_difference / (scale * wheel_vel)) 

        # Turn the robot in the correct direction
        if angle_difference < 0:
            lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        else:
            lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)

        # Update the robot's pose using SLAM
        current_pose = get_robot_pose(lv, rv, turn_time)
        x, y, th = current_pose

        while th > np.pi:
            th -= np.pi * 2
        while th <= -np.pi:
            th += np.pi * 2

        # Recalculate the angle difference
        desired_angle = np.arctan2(yg - y, xg - x)
        angle_difference = desired_angle - th

        while angle_difference > np.pi:
            angle_difference -= np.pi * 2
        while angle_difference <= -np.pi:
            angle_difference += np.pi * 2

    print(f"expected angle: {(180/math.pi)*desired_angle}\nActual angle: {(180/math.pi)*th}")
    current_pose = [x,y,th]
    return lv,rv, turn_time, current_pose
    
    

def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    #(waypoint)
    #print(robot_pose[0], robot_pose[1],(180/math.pi)*robot_pose[2])
    
    ####################################################
    # Calculate the angle to turn
    delta_x = waypoint[0] - robot_pose[0]
    delta_y = waypoint[1] - robot_pose[1]
    prev_x = robot_pose[0]
    prev_y = robot_pose[1]
    # Calculate the distance to the waypoint
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    wheel_vel = 30  # tick
    distance_error = distance
    lv,rv,drive_time = 0,0,0

    # Calculate drive time
    try:
        drive_time = distance / (wheel_vel*scale)
        if np.isnan(drive_time) or drive_time <= 0:
            raise ValueError("Invalid drive time calculated.")
    except Exception as e:
        #print(f"Error calculating drive time: {e}")
        drive_time = 1  # Set a default drive time

    #print(f"Driving for {drive_time:.2f} seconds")
    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    #robot_pose[:2] = waypoint
    current_pose = get_robot_pose(lv,rv, drive_time)

    print(f"Expected waypoint [{waypoint[0]}, {waypoint[1]}]\nActual waypoint [{current_pose[0]}, {current_pose[1]}]")
    #return robot_pose
    return lv,rv, drive_time, current_pose


def get_robot_pose(lv,rv, dt):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    drive_meas = measure.Drive(lv, -rv, dt)
    img = ppi.get_image_physical()
    aruco_detector = aruco.aruco_detector(robot)
    measurement, _ = aruco_detector.detect_marker_positions(img)
    ekf.predict(drive_meas)
    ekf.update(measurement)

    # update the robot pose [x,y,theta]
    state = ekf.get_state_vector() # replace with your calculation
    robot_pose = [state[0].item(), state[1].item(), state[2].item()]
   # print(f"Actual pose: {robot_pose}")
    ####################################################

    return robot_pose


def generate_intermediate_waypoints(start, end, interval=0.05):
    """Generate intermediate waypoints between start and end at given interval."""
    waypoints = []
    start = np.array(start)
    end = np.array(end)
    distance = np.linalg.norm(np.array(end) - np.array(start))
    num_points = int(distance // interval)
    for i in range(1, num_points + 1):
        waypoint = start + (end - start) * (i * interval / distance)
        waypoints.append(waypoint)
    # Add the final segment if it is less than the interval
    if distance % interval != 0:
        waypoints.append(end)
    return waypoints



# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='map.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

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
        x , y = aruco_true_pos[i]
        ekf.add_true_landmarks(i+1,x,y)

    # Generate obstacles
    obstacles = generate_circular_obstacles(combined_positions)

    #start = [0.0, 0.0,0.0]
    start = get_robot_pose(0,0,0)
    map_size = 2.7 #should change to about 2.6 as robot cannot touch line

    # Initialize path list
    full_path = []
    goal_indices = []
    new_full_path = []

#trying path smoothing so might be a lot faster might have more pose errror though
# Sequentially navigate to each goal
    current_position = start
    for goal in targetPose:
        while True:
            rrt = RRT(current_position[:2], goal[:2], obstacles, map_size)
            path = rrt.build_rrt()
            if path:
                next_node = path[1]  # Get the next node in the path
                expected_orientation = np.arctan2(next_node[1] - current_position[1], next_node[0] - current_position[0])
                print(f"expected pose: {next_node,expected_orientation}")
                #lv, rv, dt, current_angle = rotate_to_point(next_node, current_position,ekf,robot)
                #lv,rv,dt,current_point = drive_to_point(next_node,current_position)
                #current_position = current_point + [current_angle]
                
                #print(f'Acutal positon {current_position}, actual angle {current_angle}')

                new_full_path.append(current_position[:2])
                intermediate_waypoints = generate_intermediate_waypoints(current_position[:2],next_node)
                new_full_path.extend(intermediate_waypoints)

                #trying to split segemrnts into 10cm intervals, trying to reduce error
                #for i in range(len(intermediate_waypoints)):
                #    lv, rv, dt, current_angle = rotate_to_point(intermediate_waypoints[i], current_position,ekf,robot)
                #    lv,rv,dt,current_point = drive_to_point(intermediate_waypoints[i],current_position,ekf,robot)
                #    current_position = current_point + [current_angle]
                lv, rv, dt, current_position = rotate_to_point(intermediate_waypoints[0], current_position)
                lv,rv,dt,current_position = drive_to_point(intermediate_waypoints[0],current_position)
                #current_position = current_point + [current_angle]
                print(f'Acutal positon {current_position}')  
                position_error = np.linalg.norm(np.array(current_position[:2]) - np.array(intermediate_waypoints[0]))  

                #correction step
                #while position_error > 0.05:
                 #   print(f"Correcting position error: {position_error}")
                  #  lv, rv, dt, current_position = rotate_to_point(intermediate_waypoints[0], current_position)
                   # lv,rv,dt,current_position = drive_to_point(intermediate_waypoints[0],current_position)
                    #current_position = current_point + [current_angle]
                   # print(f'Acutal positon {current_position}')
                   # position_error = np.linalg.norm(np.array(current_position[:2]) - np.array(intermediate_waypoints[0]))  



                # Check if the robot has deviated significantly from the expected pose
                #if ekf.P[0,0] > 0.1 or ekf.P[1,1] > 0.1:
                #    for i in range(5):
                #        lv, rv = ppi.set_velocity([1, 0], tick=30, time=1)
                #        current_position = get_robot_pose(ekf,robot,lv,rv,1)

                #uncomment these for path testing
                #current_position[2] = np.arctan2(next_node[1] - current_position[1], next_node[0] - current_position[0])
                #current_position[:2] = next_node

                full_path.append(current_position.copy())
                if np.linalg.norm(np.array(current_position[:2]) - np.array(goal[:2])) < 0.3:
                    print(f"Reached goal at coordinates: {goal[:2]}")
                    #lv,rv,dt=rotate_to_face_goal(goal, current_position)
                    time.sleep(2)
                    #current_position = get_robot_pose(ekf,robot,lv,rv,dt)
                    goal_indices.append(len(full_path)) 
                    break
            else:
                print(f"No path found to goal {goal[:2]} going back to origin")
                # trace back to origin and start again if cannot find path
                for node in reversed(full_path):
                    #lv, rv, dt = rotate_to_point(node[:2], current_position)
                    #current_position = get_robot_pose(ekf,robot,lv,rv,dt)  # Update the robot's pose theta
                    #lv, rv, dt = drive_to_point(node[:2], current_position)
                    #current_position = get_robot_pose(ekf,robot,lv,rv,dt)  # Update the robot's pose x and y
                    print(f"Tracing back to: {node[:2]}")
            # Retry reaching the goal
                current_position = start
                full_path = [start]
                #break
                

    if len(full_path) > 0:
        print("Full path found!")
        #for point in full_path:
        #    print(point)
    else:
        print("No full path found.")

            # Ensure all waypoints have the same length (x, y, theta)
new_full_path = [(point[0], point[1], 0.0) if len(point) == 2 else point for point in new_full_path]

    # Convert to numpy array
new_full_path = np.array(new_full_path)

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
                ax.text(obstacle.center[0], obstacle.center[1], str(i+1), color='white', ha='center', va='center')
                break
    else:
        circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, color='gray')
        ax.add_patch(circle)

if len(full_path) > 0:
    new_full_path = [(0.0,0.0,0.0)] + new_full_path
    new_full_path =  np.array(new_full_path)
    ax.plot(new_full_path[:, 0], new_full_path[:, 1], '-o', color='blue')  # Plot the full path as a continuous line

    # Plot tiny black arrows to visualize orientation
    for x, y, theta in new_full_path:
        dx = 0.1 * np.cos(theta)  # Scale the arrow length as needed
        dy = 0.1 * np.sin(theta)
        ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='black')

plt.gca().invert_xaxis()  # Invert x-axis
plt.gca().invert_yaxis()  # Invert y-axis
plt.show()

    

