# Robot Teleoperation with SLAM and Object Detection

import os
import sys
import time
import cv2
import numpy as np
import random
import pygame
import shutil
import argparse

# Add utility directories to the system path
sys.path.insert(0, os.path.join(os.getcwd(), "util"))
from util.pibot import PenguinPi  # Robot control class
import util.DatasetHandler as dh   # Data handling utilities
import util.measure as measure     # Measurement utilities
from util.operate_util import *
from util.mapping_util import *

sys.path.insert(0, os.path.join(os.getcwd(), "slam"))
from slam.ekf import EKF           # Extended Kalman Filter for SLAM
from slam.robot import Robot       # Robot model
import slam.aruco_detector_center as aruco  # ArUco marker detection

sys.path.insert(0, os.path.join(os.getcwd(), "path_planning"))
from path_planning.RRTC import RRTC         # Rapidly-exploring Random Tree (RRT) algorithm
from path_planning.Obstacle import *        # Obstacle definitions

sys.path.insert(0, os.path.join(os.getcwd(), "YOLO"))
from YOLO.detector import Detector  # YOLO object detector

class RobotOperation:
    def __init__(self, args):
        # Load maps and target lists
        self.fruit_list, self.fruit_positions, self.aruco_positions = read_lab_output(
            f'lab_output/{args.slam}', f'lab_output/{args.targets}')
        self.shopping_list = read_search_list(args.search_list)

        # Setup data directories
        self.data_folder = 'pibot_dataset/'
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        else:
            shutil.rmtree(self.data_folder)
            os.makedirs(self.data_folder)

        # Initialize robot
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # Initialize SLAM components
        self.ekf = self.initialize_ekf(args.calib_dir, args.ip, self.aruco_positions)
        self.aruco_detector = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)

        # Data recording setup
        self.data_writer = dh.DatasetWriter('record') if args.save_data else None
        self.output_writer = dh.OutputWriter('lab_output')

        # Command and control variables
        self.command = {'motion': [0, 0], 'inference': False, 'output': False,
                        'save_inference': False, 'save_image': False}
        self.quit_program = False
        self.ekf_active = True
        self.notification = 'Starting Robot Operation'
        self.image_counter = 0
        self.count_down = 3000
        self.start_time = time.time()
        self.control_timer = time.time()

        # Image placeholders
        self.current_image = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_image = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detection_output = np.zeros([240, 320], dtype=np.uint8)

        # Initialize YOLO detector
        if args.yolo_model:
            self.detector = Detector(args.yolo_model)
            self.detection_visualization = np.ones((240, 320, 3)) * 100
        else:
            self.detector = None
            self.detection_visualization = cv2.imread('pics/8bit/detector_splash.png')

        # GUI background
        self.gui_background = pygame.image.load('pics/gui_mask.jpg')

        # Navigation variables
        self.navigation_complete = False
        self.action_queue = []
        self.closest_distance = float('inf')
        self.stop_duration = time.time()
        self.is_stopping = False
        self.path_checkpoints = []

    def plan_navigation(self):
        """
        Plans the path to navigate to all fruits in the shopping list.
        Populates self.path_checkpoints with lists of waypoints.
        """
        initial_position = [0, 0]
        full_obstacle_list = self.aruco_positions.tolist() + self.fruit_positions.tolist()
        safe_radii = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1]

        # Generate waypoints for each fruit in the shopping list
        for fruit_index, fruit_name in enumerate(self.shopping_list):
            target_position = self.fruit_positions[self.fruit_list.index(fruit_name)].tolist()
            temp_obstacles = full_obstacle_list.copy()
            temp_obstacles.remove(target_position)

            for radius in safe_radii:
                obstacles = create_obstacle_list(temp_obstacles, radius)
                possible_routes = []

                # Try multiple times to find a valid route
                for _ in range(150):
                    start_position = self.path_checkpoints[-1][-1] if self.path_checkpoints else initial_position
                    rrt = RRTC(
                        start=start_position,
                        goal=target_position,
                        obstacle_list=obstacles,
                        width=2.7,
                        height=2.7,
                        expand_dis=0.2,
                        path_resolution=0.01,
                        max_points=1000
                    )
                    path = rrt.planning()
                    if path:
                        possible_routes.append(path[1:-1])

                if possible_routes:
                    shortest_route = min(possible_routes, key=length_of_path)
                    self.path_checkpoints.append(shortest_route)
                    print(f"Path to {fruit_name} found with safety radius {radius}")
                    break

    def execute_next_action(self):
        """
        Converts the next checkpoint into actionable commands.
        """
        if not self.path_checkpoints:
            self.navigation_complete = True
            return False
        else:
            current_target = self.shopping_list[len(self.shopping_list) - len(self.path_checkpoints)]
            self.notification = f"Navigating to {current_target}"

        if not self.path_checkpoints[0]:
            self.path_checkpoints.pop(0)
            return self.execute_next_action()

        next_point = self.path_checkpoints[0].pop(0)
        self.prepare_motion_commands(next_point)
        return True

    def prepare_motion_commands(self, waypoint):
        """
        Prepares motion commands to reach the specified waypoint.
        """
        robot_state = self.get_robot_state()
        delta_x = waypoint[0] - robot_state[0]
        delta_y = waypoint[1] - robot_state[1]
        target_angle = np.arctan2(delta_y, delta_x)

        # Add rotation command to face the target
        self.action_queue.append({'type': 'rotate', 'angle': target_angle})

        # Add a brief stop
        self.action_queue.append({'type': 'stop', 'duration': 0.5})

        # Add movement command to reach the waypoint
        self.action_queue.append({'type': 'move', 'position': waypoint})

    def perform_actions(self):
        """
        Executes actions from the action queue.
        """
        if not self.action_queue:
            if self.execute_next_action():
                self.perform_actions()
            return

        current_action = self.action_queue[0]
        action_type = current_action['type']

        if action_type == 'rotate':
            self.handle_rotation(current_action)
        elif action_type == 'move':
            self.handle_movement(current_action)
        elif action_type == 'stop':
            self.handle_stop(current_action)
        else:
            print(f"Unknown action type: {action_type}")
            self.action_queue.pop(0)

    def handle_rotation(self, action):
        """
        Handles rotation to a target angle.
        """
        current_theta = self.get_robot_state()[2]
        angle_error = ang_clamp(action['angle'] - current_theta)

        if abs(angle_error) <= 0.1:
            self.action_queue.pop(0)
        else:
            self.command['motion'] = [0, -1] if angle_error <= 0 else [0, 1]

    def handle_movement(self, action):
        """
        Handles movement to a target position.
        """
        distance_to_target = self.calculate_distance(action['position'])
        if distance_to_target < self.closest_distance and distance_to_target >= 0.01:
            self.closest_distance = distance_to_target
            self.command['motion'] = [1, 0]
        else:
            self.command['motion'] = [0, 0]
            self.closest_distance = float('inf')
            self.action_queue.pop(0)

    def handle_stop(self, action):
        """
        Handles stopping for a specified duration.
        """
        if time.time() > self.stop_duration and self.is_stopping:
            self.action_queue.pop(0)
            self.is_stopping = False
        else:
            self.command['motion'] = [0, 0]
            if not self.is_stopping:
                self.stop_duration = time.time() + action['duration']
                self.is_stopping = True

    def calculate_distance(self, waypoint):
        """
        Calculates the distance from the robot to the waypoint.
        """
        robot_state = self.get_robot_state()
        delta_x = waypoint[0] - robot_state[0]
        delta_y = waypoint[1] - robot_state[1]
        return np.hypot(delta_x, delta_y)

    def control_motors(self):
        """
        Sends motion commands to the robot and records drive measurements.
        """
        if args.play_data:
            left_vel, right_vel = self.pibot.set_velocity()
        else:
            left_vel, right_vel = self.pibot.set_velocity(self.command['motion'])

        if self.data_writer:
            self.data_writer.write_keyboard(left_vel, right_vel)

        time_delta = time.time() - self.control_timer
        drive_measurement = measure.Drive(left_vel, -right_vel, time_delta) if args.ip != 'localhost' else measure.Drive(left_vel, right_vel, time_delta)
        self.control_timer = time.time()
        return drive_measurement

    def capture_image(self):
        """
        Captures an image from the robot's camera.
        """
        self.current_image = self.pibot.get_image()
        if self.data_writer:
            self.data_writer.write_image(self.current_image)

    def update_slam(self, drive_measurement):
        """
        Updates the SLAM state with new drive measurements and marker observations.
        """
        markers_detected, self.aruco_image = self.aruco_detector.detect_marker_positions(self.current_image)
        if self.ekf_active:
            self.ekf.predict(drive_measurement)
            self.ekf.add_landmarks(markers_detected)
            self.ekf.update(markers_detected)

    def run_object_detection(self):
        """
        Runs the YOLO object detector on the current image.
        """
        if self.command['inference'] and self.detector:
            input_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            self.detection_output, self.detection_visualization = self.detector.detect_single_image(input_image)
            self.detection_visualization = cv2.cvtColor(self.detection_visualization, cv2.COLOR_RGB2BGR)
            self.command['inference'] = False

    def save_current_image(self):
        """
        Saves the current image to the data folder.
        """
        if self.command['save_image']:
            filename = os.path.join(self.data_folder, f'img_{self.image_counter}.png')
            image_to_save = cv2.cvtColor(self.pibot.get_image(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, image_to_save)
            self.image_counter += 1
            self.command['save_image'] = False
            self.notification = f'Image saved as {filename}'

    def initialize_ekf(self, calib_dir, ip, known_aruco_positions):
        """
        Initializes the EKF with calibration parameters.
        """
        camera_matrix = np.loadtxt(os.path.join(calib_dir, "intrinsic.txt"), delimiter=',')
        dist_coeffs = np.loadtxt(os.path.join(calib_dir, "distCoeffs.txt"), delimiter=',')
        scale = np.loadtxt(os.path.join(calib_dir, "scale.txt"), delimiter=',')
        baseline = np.loadtxt(os.path.join(calib_dir, "baseline.txt"), delimiter=',')

        if ip == 'localhost':
            scale /= 2

        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot, known_aruco_positions=known_aruco_positions)

    def get_robot_state(self):
        """
        Retrieves the current state of the robot.
        """
        return self.ekf.robot.state

    def render_gui(self, canvas):
        """
        Renders the GUI components on the canvas.
        """
        canvas.blit(self.gui_background, (0, 0))
        vertical_padding = 40
        horizontal_padding = 30

        # SLAM Visualization
        slam_view = self.ekf.draw_slam_state(
            self.fruit_positions, self.fruit_list,
            checkpoints=self.path_checkpoints, shopping_list=self.shopping_list,
            res=(600, 600), not_pause=self.ekf_active)
        canvas.blit(slam_view, (2 * horizontal_padding + 320, vertical_padding))
        self.draw_pygame_image(canvas, self.aruco_image, position=(horizontal_padding, vertical_padding))

        # Object Detection Visualization
        detection_view = cv2.resize(self.detection_visualization, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_image(canvas, detection_view, position=(horizontal_padding, 240 + 2 * vertical_padding))

        # Add captions and notifications
        self.add_caption(canvas, 'SLAM', position=(2 * horizontal_padding + 320, vertical_padding))
        self.add_caption(canvas, 'Detector', position=(horizontal_padding, 240 + 2 * vertical_padding))
        self.add_caption(canvas, 'PiBot Cam', position=(horizontal_padding, vertical_padding))

        notification_surface = TEXT_FONT.render(self.notification, False, (220, 220, 220))
        canvas.blit(notification_surface, (horizontal_padding + 10, 596))

        # Countdown timer
        remaining_time = self.count_down - (time.time() - self.start_time)
        time_display = f'Count Down: {int(remaining_time)}s' if remaining_time > 0 else "Time Is Up !!!"
        countdown_surface = TEXT_FONT.render(time_display, False, (50, 50, 50))
        canvas.blit(countdown_surface, (2 * horizontal_padding + 320 + 5, 530))

        return canvas

    @staticmethod
    def draw_pygame_image(canvas, image, position):
        """
        Converts and draws an OpenCV image onto the pygame canvas.
        """
        image = np.rot90(image)
        surface = pygame.surfarray.make_surface(image)
        surface = pygame.transform.flip(surface, True, False)
        canvas.blit(surface, position)

    @staticmethod
    def add_caption(canvas, text, position):
        """
        Adds a caption to the pygame canvas.
        """
        caption_surface = TITLE_FONT.render(text, False, (200, 200, 200))
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    def handle_keyboard_input(self):
        """
        Processes keyboard inputs for robot control.
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # Movement commands
                if event.key == pygame.K_UP:
                    self.command['motion'][0] = min(self.command['motion'][0] + 1, 1)
                elif event.key == pygame.K_DOWN:
                    self.command['motion'][0] = max(self.command['motion'][0] - 1, -1)
                elif event.key == pygame.K_LEFT:
                    self.command['motion'][1] = min(self.command['motion'][1] + 1, 1)
                elif event.key == pygame.K_RIGHT:
                    self.command['motion'][1] = max(self.command['motion'][1] - 1, -1)
                elif event.key == pygame.K_SPACE:
                    self.command['motion'] = [0, 0]
                # Other commands
                elif event.key == pygame.K_i:
                    self.command['save_image'] = True
                elif event.key == pygame.K_p:
                    self.command['inference'] = True
                elif event.key == pygame.K_ESCAPE:
                    self.quit_program = True
            elif event.type == pygame.QUIT:
                self.quit_program = True

        if self.quit_program:
            self.command['motion'] = [0, 0]
            self.control_motors()
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='192.168.50.1')
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--slam", type=str, default='slam.txt')
    parser.add_argument("--targets", type=str, default='targets.txt')
    parser.add_argument("--search_list", type=str, default="shopping_list.txt")
    parser.add_argument("--yolo_model", type=str, default='YOLO/model/yolov8_model.pt')
    args = parser.parse_args()

    # Initialize pygame and fonts
    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    # Setup display
    screen_width, screen_height = 980, 780
    canvas = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Robot Operation GUI')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash_screen = pygame.image.load('pics/loading.png')
    pygame.display.update()

    # Splash screen
    start_program = False
    while not start_program:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start_program = True
        canvas.blit(splash_screen, (0, 0))
        pygame.display.update()

    # Initialize robot operation
    robot_operation = RobotOperation(args)
    robot_operation.plan_navigation()
    input("Press Enter to start the robot operation...")

    # Main loop
    while not robot_operation.navigation_complete:
        robot_operation.handle_keyboard_input()
        if not robot_operation.quit_program:
            robot_operation.perform_actions()
        else:
            robot_operation.command['motion'] = [0, 0]

        robot_operation.capture_image()
        drive_measurement = robot_operation.control_motors()
        robot_operation.update_slam(drive_measurement)
        robot_operation.save_current_image()
        robot_operation.run_object_detection()

        # Render GUI
        robot_operation.render_gui(canvas)
        pygame.display.update()