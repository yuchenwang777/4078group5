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

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure


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
                target_positions.append(-1*fruit_true_pos[i])
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
    #(waypoint)
    #print(robot_pose[0], robot_pose[1],(180/math.pi)*robot_pose[2])
    
    ####################################################
    # Calculate the angle to turn
    xg, yg = waypoint
    x,y,th = robot_pose
    
    
    # Normalize the angle to be within the range [-pi, pi]
    #angle = (angle + np.pi) % (2 * np.pi) - np.pi
    #print((180/math.pi)*angle)
    desired_angle = np.arctan2(yg - y, xg - x)
    current_angle = th
    print((180/math.pi)*desired_angle)
    angle_difference = desired_angle - current_angle
    print((180/math.pi)*angle_difference)
    while angle_difference>np.pi:
        angle_difference-=np.pi*2
    while angle_difference<=-np.pi:
        angle_difference+=np.pi*2

    wheel_vel = 30  # tick
    
    # Calculate turn time
    turn_time=abs(baseline*angle_difference*0.5/(scale*wheel_vel))
    print(f"Turning for {turn_time:.2f} seconds")
    robot_pose[2] = desired_angle
    if angle_difference < 0:
        print("turning left")
        ppi.set_velocity([0, -1],turning_tick=wheel_vel, time=turn_time)
    else:
        print("turning right")
        ppi.set_velocity([0, 1],turning_tick=wheel_vel, time=turn_time)

    #drive_to_point(waypoint,)
    #time.sleep(turn_time)
    return robot_pose
    
    

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
    # Calculate the distance to the waypoint
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    wheel_vel = 50  # tick

    # Calculate drive time
    try:
        drive_time = distance / (wheel_vel*scale)
        if np.isnan(drive_time) or drive_time <= 0:
            raise ValueError("Invalid drive time calculated.")
    except Exception as e:
        print(f"Error calculating drive time: {e}")
        drive_time = 1  # Set a default drive time

    print(f"Driving for {drive_time:.2f} seconds")
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    robot_pose[:2] = waypoint
    
    ####################################################

    print(f"Arrived at [{waypoint[0]}, {waypoint[1]}]")

    return robot_pose


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
    # Get the dimensions of the image
    img_width, img_height = map_image.size
    
    # Convert pixel coordinates to the desired range
    x = (event.x / img_width) * 3 - 1.5
    y = (event.y / img_height) * 3 - 1.5
    
    # based on the direction of your x/y axis my need to adjust x and y (+/-) to get the correct orientation
    # Invert x-axis to match typical coordinate system
    x = -x
    robot_pose = rotate_to_point([x, y], robot_pose)
    robot_pose = drive_to_point([x,y],robot_pose)
    print(f"Clicked coordinates: ({x:.2f}, {y:.2f})")
    
    # Update the robot pose
    #this was just for testing GUI need to use control algorithm to update the robot pose
    # call drive_to_point function here and use x,y as inputs, along with current robot pose
    #theta = math.atan2(robot_pose[1]-y,  robot_pose[0]-x)
    #robot_pose = [x, y, theta] 
    print(f"Updated robot pose: [{robot_pose[0]}, {robot_pose[1]}, {(180/math.pi)*robot_pose[2]}]")
    
    # Redraw the map image to clear previous circles
    map_image_copy = map_image.copy()
    draw_robot(event.x, event.y, robot_pose[2])

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
    orientation = robot_pose[2] +math.pi
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

        radius = 10  # Radius of the fruit
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='green')
    
    # Draw the ArUco markers
    for i,pos in enumerate(aruco_true_pos):
        x = int((1.5 - pos[0]) / 3 * width)
        y = int((pos[1] + 1.5) / 3 * height)
        side = 20  # Side length of the square
        #drawing black blocks for markers
        draw.rectangle((x - side // 2, y - side // 2, x + side // 2, y + side // 2), fill='black')
        #numbering the blocks
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
    targetPose = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

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

