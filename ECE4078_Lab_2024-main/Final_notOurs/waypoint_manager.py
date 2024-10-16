from slam.robot import Robot
import time 
import numpy as np
from util.clear_out import clear

class wp_manager:
     
    def __init__(self):
        
        self.wps = []
        self.current_wp = 0 # index for self.wps
        self.completed_route = False
        self.end_timer = 0
        self.end_pos = (0, 0)
        self.arrived = False
        self.state = "Pause"
        self.finished_waiting = False
        
        self.desired_heading = ''
        self.distance_to_wp = ''
        self.distance_to_goal = ''

        self.K_pv = 4.0
        self.K_pw = 2.0 

        self.seen_last_landmark_threshhold = 10
        self.looking_at_landmark_threshhold = 3
        self.seen_last_landmark_timer = time.time()
        self.looking_at_landmark_timer = time.time()
        self.b_looking_for_lm = False
        

    def tick(self, robot, b_path_found, b_saw_lm_last_tick, lms):
        
        ############################################################################
        ############################################################################
        ###
        ###     Inputs:
        ###          - way_points : a list of way points    [ (x1, y1), (x2, y2), ... ]
        ###          - Robot : robot object, robot.state = [x, y, th]
        ###     Outputs:
        ###          - Motion array:  [v, w]
        ###
        ###########################################################################
        ###########################################################################
        #print(self.state)
        # Turn to look at landmark if you havent seen one in a while
        '''
        if b_saw_lm_last_tick:
            self.seen_last_landmark_timer = time.time()
            if self.b_looking_for_lm:
                if time.time() - self.looking_at_landmark_timer < self.looking_at_landmark_threshhold:
                    self.state = "Looking at LM"
                    return [0, 0]
                else:
                    self.b_looking_for_lm = False
        else:
            self.looking_at_landmark_timer = time.time()
            if time.time() - self.seen_last_landmark_timer > 30:
                return [0, 1]
            elif time.time() - self.seen_last_landmark_timer > self.seen_last_landmark_threshhold:
                self.b_looking_for_lm = True
                self.state = "searching for LM"
                desired_heading = self.get_direction_to_nearest_lm(robot, lms)
                speed = max(2, min(1, 5*desired_heading))
                if desired_heading < 0:
                    return [0, -speed]
                else:
                    return [0, speed]
        '''

        if not b_path_found:
            self.state = "Path not found"
            return [0, 1]

        if len(self.wps) == 0:
            return [0, 0]

        # Pause after End
        if self.arrived:
            if time.time() - self.end_timer < 2.0:
                self.state = "Pause near goal"
                return [0, 0]
            else:
                self.completed_route = True
                self.arrived = False
                return [0, 0]

        if self.current_wp >= len(self.wps):
            self.state = "Finished :D"
            return [0, 0]
        
        
        end_pos = np.array([self.end_pos[0], self.end_pos[1]])
        self.distance_to_goal = self.get_distance_robot_to_goal(robot.state, end_pos)
        self.distance_to_wp = self.get_distance_robot_to_goal(robot.state, self.wps[self.current_wp])
        self.desired_heading = self.get_angle_robot_to_goal(robot.state, self.wps[self.current_wp])
        stop_criteria_met =  np.sqrt(abs(robot.state[0] - self.wps[self.current_wp][0])**2 + abs(robot.state[1] - self.wps[self.current_wp][1])**2) < 0.03
        #print(self.distance_to_goal, self.desired_heading*180/np.pi)

        dist_from_final_goal = np.sqrt(abs(robot.state[0] - self.end_pos[0])**2 + abs(robot.state[1] - self.end_pos[1])**2)
        if (dist_from_final_goal < 0.3) or (self.finished_waiting):
            self.end_timer = time.time()
            self.finished_waiting = False
            self.arrived = True
            self.current_wp += 1
            self.state = "Begin pause 2 seconds"
            return [0, 0]
        
        if stop_criteria_met:
            self.forward_timer = time.time()
            if self.current_wp+1 < len(self.wps):
                self.current_wp += 1
            else:
                self.finished_waiting = True
            self.state = "Pause after reaching way point"
            return [0, 0]
        
        # First turn to destination
        if abs(self.desired_heading) > 0.12:
            self.state = "Turn"
            if (self.desired_heading < 0):
                return [0, -self.K_pw]
            else:
                return [0, self.K_pw]
        
        
        # Make sure to pause before moving forward
        if self.state == "Turn":
            self.turn_timer = time.time()
            self.state = "Pause after turn"
            return [0, 0]

        # Second Move forward to
        self.state = "Move"
        return [self.K_pv, 0]

    def reset(self, new_wps, new_end_pos):
        self.arrived = False
        self.current_wp = 0
        self.end_pos = new_end_pos
        self.wps = new_wps
        self.completed_route = False

    def get_direction_to_nearest_lm(self, robot, lms):
        max_dist = 0
        closest_lm_pos = np.array([lms[0][0], lms[1][0]])
        for i in range(len(lms[0])):
            dist = self.get_distance_robot_to_goal(robot.state, np.array([lms[0][i], lms[1][i]]))
            if dist > max_dist:
                closest_lm_pos = np.array([lms[0][i], lms[1][i]])
                max_dist = dist
        return self.get_angle_robot_to_goal(robot.state, closest_lm_pos)

    def get_angle_robot_to_goal(self, robot_state, goal):
        if goal.shape[0] < 3:
            goal = np.hstack((goal, np.array([0])))
        x_goal, y_goal,_ = goal
        x, y, theta = robot_state
        x_diff = x_goal - x
        y_diff = y_goal - y
        alpha = self.clamp_angle(np.arctan2(y_diff, x_diff) - theta)
        return alpha

    def clamp_angle(self, rad_angle=0, min_value=-np.pi, max_value=np.pi):
        if min_value > 0:
            min_value *= -1
        angle = (rad_angle + max_value) % (2 * np.pi) + min_value
        return angle

    @staticmethod
    def get_distance_robot_to_goal(robot_state, goal):
        if goal.shape[0] < 3:
            goal = np.hstack((goal, np.array([0])))
            x_goal, y_goal,_ = goal
            x, y,_ = robot_state
            x_diff = x_goal - x
            y_diff = y_goal - y
            rho = np.hypot(x_diff, y_diff)
            return rho
        
    