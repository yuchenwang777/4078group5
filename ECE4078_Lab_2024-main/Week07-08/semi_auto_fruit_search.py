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
    # Get the dimensions of the image
    img_width, img_height = map_image.size
    
    # Convert pixel coordinates to the desired range
    x = (event.x / img_width) * 3 - 1.5
    y = (event.y / img_height) * 3 - 1.5
    
    # Invert y-axis to match typical coordinate system
    x = -x
    
    print(f"Clicked coordinates: ({x:.2f}, {y:.2f})")
    
    # Update the robot pose
    #this was just for testing GUI need to use control algorithm to update the robot pose
    # call drive_to_point function here and use x,y as inputs, along with current robot pose
    robot_pose = [x, y, 0.0] 
    print(f"Updated robot pose: {robot_pose}")
    
    # Redraw the map image to clear previous circles
    map_image_copy = map_image.copy()
    draw_robot(event.x, event.y)

def draw_target_circles(image, targets):
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    for target in targets:
        # Convert target coordinates to pixel coordinates
        x = int((target[0] + 1.5) / 3 * img_width)
        y = int((1.5 - target[1]) / 3 * img_height)
        
        # Draw a circle with a 0.5m radius around the target
        radius = int(0.5 / 3 * img_width)  # Convert 0.5m to pixel radius
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline='blue', width=3)  # Increase width for better visibility

def draw_robot(x, y):
    global map_image_copy
    # Draw a red circle on the image at the specified coordinates
    draw = ImageDraw.Draw(map_image_copy)
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
    
    # Update the image in the label
    map_photo.paste(map_image_copy)

def process_waypoints(x, y):
    # Convert pixel coordinates to actual waypoints if needed
    # For now, just print them
    print(f"Waypoint coordinates: ({x}, {y})")
    # estimate the robot's pose
    robot_pose = get_robot_pose()
    print(f"Robot pose: {robot_pose}")

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

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # Create the main window
    root = tk.Tk()
    root.title("Map Click Waypoints")

    # Load the map image
    map_image = Image.open("M4_prac_map_layout_cropped.png")  # Update with your map image path
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

    print(f"Final robot pose: {robot_pose}")

        # robot drives to the waypoint
       # waypoint = [x,y]
        #drive_to_point(waypoint,robot_pose)
        #print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        #ppi.set_velocity([0, 0])
        #uInput = input("Add a new waypoint? [Y/N]")
        #if uInput == 'N':
        #    break

