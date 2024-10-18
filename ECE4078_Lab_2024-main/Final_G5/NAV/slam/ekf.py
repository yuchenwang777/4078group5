import numpy as np
import cv2
import pygame
from util.fruit_search_help import is_shopping_fruit
class EKF:
    # The state is ordered as [x; y; theta; l1x; l1y; ...; lnx; lny]

    def __init__(self, robot, ekf_state_change_threshhold = 0.001):
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))
        self.taglist = []

        # LM Covariance matrix
        self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3
        #self.init_lm_cov_known = 1e-31
        #self.init_lm_cov_known = 1e3**2
        self.init_lm_cov_known = 1e-32
        self.robot_init_state = None
        self.pibot_pic = pygame.image.load(f'./pics/8bit/pibot_top.png')

        self.lm_pics = []
        for i in range(1, 11):
            f_ = f'./pics/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./pics/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))

        self.class_names = ['capsicum', 'garlic', 'lemon', 'lime', 'pear', 'potato', 'pumpkin', 'tomato']
        self.fruit_pics = {}
        self.fruit_taglist = []
        for fruit_name in self.class_names:
            f_fruit_ = f'./pics/8bit_fruit/{fruit_name}.png'
            self.fruit_pics[fruit_name] = pygame.image.load(f_fruit_)
        f_fruit_ = f'./pics/8bit_fruit/unknown.png'
        self.fruit_pics["Uknown"] = pygame.image.load(f_fruit_)
        self.is_first_iter = True
        self.m2pixel = 150
        self.state_change_threshhold = ekf_state_change_threshhold
            

    def reset(self):
        self.robot.state = np.zeros((3, 1))
        self.markers = np.zeros((2,0))
        self.taglist = []
        # Covariance matrix
        self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3
        self.robot_init_state = None

    def number_landmarks(self):
        return int(self.markers.shape[1])
    
    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state

    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')
        
    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    def scale_landmark_pos(self, lms):
        # Calibrate distance according to camera immage using regression
        for i in range(len(lms)):
            lm = lms[i]
            # calibrated x pos
            lm.position[0][0] = (lm.position[0][0] + 0.01743)/1.062
            # calibrated y pos
            lm.position[1][0] = (lm.position[1][0]*0.9536)-0.01272
            
        return lms
    
    def get_centre_pos(self, lms, rvecs, tvecs, ids):
        
        for i in range(len(lms)):
            lms[i].position[0][0] = lms[i].position[0][0] + 0.04
        return lms

    def remove_unknown_lms(self, lms):
        new_lms = []
        for lm in lms:
            if lm.tag in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                new_lms.append(lm)
        return new_lms

    def change_landmark_covar(self, lms):
        state = self.robot.state
        new_lms = []
        for i in range(len(lms)):
            lm = lms[i]
            lm_x, lm_y = lm.position[0][0], lm.position[1][0]
            dist = np.sqrt(lm_x**2 + lm_y**2)
            if dist < 1.5:
                lm.covariance = (np.eye(2)) * ((np.exp(dist/1.3)-1)/5)
                new_lms.append(lm)
        return new_lms  

    # the prediction step of EKF
    def predict(self, raw_drive_meas):
        A = self.state_transition(raw_drive_meas)
        Q = self.predict_covariance(raw_drive_meas)
        self.robot.drive(raw_drive_meas)
        self.P = A@self.P@np.transpose(A) + Q

    # the update step of EKF
    def update(self, measurements):

        if not measurements:
            return False, 0

        # Construct measurement index list
        tags = [lm.tag for lm in measurements]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements], axis=0)
        R = np.zeros((2*len(measurements),2*len(measurements)))

        for i in range(len(measurements)):
            if not self.is_first_iter:
                R[2*i:2*i+2,2*i:2*i+2] = measurements[i].covariance

        # Compute own measurements
        z_hat = self.robot.measure(self.markers, idx_list)
        z_hat = z_hat.reshape((-1,1),order="F")
        C = self.robot.derivative_measure(self.markers, idx_list)

        # Correcting / updating the estimate of the current state with observations from sensor readings 
        x_bar = self.get_state_vector()
        K = self.P@C.T@np.linalg.inv(C@self.P@C.T + R)
        W = K@(z - z_hat)
        corrected_x = x_bar + W
        corrected_P = (np.identity(K.shape[0]) - K@C)@self.P

        b_state_change_threshhold, average = self.check_state_change_thresh(W)

        self.P = corrected_P 
        self.set_state_vector(corrected_x)
        self.is_first_iter = False
        return b_state_change_threshhold, average

    def state_transition(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(raw_drive_meas)
        return F
    
    def predict_covariance(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n)) 
        # do we need drift for M4? 0.001
        if not self.is_first_iter:
            #if raw_drive_meas.left_speed > 0.0001 or raw_drive_meas.left_speed > 0.0001:
            Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas)+ 0.001*np.eye(3)
            #Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas)+ 1e-30*np.eye(3)
        return Q

    def add_landmarks(self, measurements, known_lm=False):
        if not measurements:
            return

        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        # Add new landmarks to the state
        for lm in measurements:
            if lm.tag in self.taglist:
                # ignore known tags
                continue

            if not known_lm:

                lm_bff = lm.position
                
                lm_inertial = robot_xy + R_theta @ lm_bff
                
                self.taglist.append(int(lm.tag))
                self.markers = np.concatenate((self.markers, lm_inertial), axis=1)

                # Create a simple, large covariance to be fixed by the update step
                self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
                self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)

                self.P[-2,-2] = self.init_lm_cov**2
                self.P[-1,-1] = self.init_lm_cov**2
                
            else:
                self.taglist.append(int(lm.tag))
                self.markers = np.concatenate((self.markers, lm.position), axis=1)
                self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
                self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
                self.P[-2,-2] = self.init_lm_cov_known
                self.P[-1,-1] = self.init_lm_cov_known

    def check_state_change_thresh(self, w):

        average = 0
        for x in w:
            average += x[0]
        average = abs(average / len(w))
        #print("EKF CHANGE AVERAGE: ", average)
        if average > self.state_change_threshhold:
            return True, average
        return False, average
    
    def draw_slam_state(self, fruit_true_pos=None, fruit_list=None,checkpoints=None,shopping_list = None, res=(600, 600), not_pause=True):        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy*0
        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
        
        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (0, 30, 56), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (244, 69, 96), 1)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
                    (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1],
                    (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1],
                    (coor_[0]-5, coor_[1]-5))
        return surface

    # def draw_slam_state(self, res, wp_manager, grocery_idx, shopping_list, fruit_pose_dict, colors):
    #     bg_rgb = np.array([colors.COLOR_SLAM_BG]).reshape(1, 1, 3)  
    #     canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)

    #     # in meters, 
    #     lms_xy = self.markers[:2, :]
    #     robot_xy = self.robot.state[:2, 0].reshape((2, 1))
    #     lms_xy = lms_xy - robot_xy

    #     # initialise robot coordinates 
    #     robot_theta = self.robot.state[2,0]
    #     start_point_uv = self.to_im_coor((0, 0), res, self.m2pixel)
        
    #     p_robot = self.P[0:2,0:2]
    #     axes_len, angle = self.make_ellipse(p_robot)
    #     '''
    #     canvas = cv2.ellipse(canvas, start_point_uv, 
    #                 (int(axes_len[0]*self.m2pixel), int(axes_len[1]*self.m2pixel)),
    #                 angle, 0, 360, (0, 30, 56), 1)
    #     '''
    #     canvas = cv2.ellipse(   img=canvas, 
    #                             center=tuple(map(int, start_point_uv)),
    #                             axes=(int(axes_len[0]*self.m2pixel), int(axes_len[1]*self.m2pixel)),
    #                             angle=int(angle), 
    #                             startAngle=0,
    #                             endAngle=360, 
    #                             color= colors.COLOR_ROBOT_COVAR,
    #                             thickness=1)
    #     # draw landmards
    #     if self.number_landmarks() > 0:
    #         for i in range(len(self.markers[0,:])):
    #             xy = (lms_xy[0, i], lms_xy[1, i])
    #             coor_ = self.to_im_coor(xy, res, self.m2pixel)
    #             # plot covariance
    #             Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
    #             axes_len, angle = self.make_ellipse(Plmi)
    #             canvas = cv2.ellipse(canvas, coor_, 
    #                 (int(axes_len[0]*self.m2pixel), int(axes_len[1]*self.m2pixel)),
    #                 angle, 0, 360, colors.COLOR_LM_COVAR, 1)

    #     surface = pygame.surfarray.make_surface(np.rot90(canvas))
    #     surface = pygame.transform.flip(surface, True, False)

    #     # draw grid with centre at  middle of robot = (0, 0)
    #     ekf_view_width, ekf_view_height = res
    #     num_lines = 60

    #     robot_x = self.robot.state[:2, 0][0]
    #     robot_y = self.robot.state[:2, 0][1]

    #     # vertical lines
    #     for x_i in range(-int(num_lines/2), int(num_lines/2), 2):
    #         x = x_i / 10
    #         xy_start = (x - robot_x, - ekf_view_height - robot_y)
    #         xy_end = (x - robot_x, ekf_view_height - robot_y)
            
    #         coor_start = self.to_im_coor(xy_start, res, self.m2pixel)
    #         coor_end = self.to_im_coor(xy_end, res, self.m2pixel)

    #         pygame.draw.line(surface, colors.COLOR_SLAM_GRID, coor_start, coor_end, 2)

    #     # horizontal lines
    #     for y_i in range(-int(num_lines/2), int(num_lines/2), 2):
    #         y = y_i / 10
    #         xy_start = (-ekf_view_width - robot_x, y - robot_y)  # Start at the far left (negative x)
    #         xy_end = (ekf_view_width - robot_x, y - robot_y)
            
    #         coor_start = self.to_im_coor(xy_start, res, self.m2pixel)
    #         coor_end = self.to_im_coor(xy_end, res, self.m2pixel)

    #         pygame.draw.line(surface, colors.COLOR_SLAM_GRID, coor_start, coor_end, 2)

    #     for x in [-1.5, 1.5]:
    #         xy_start = (x - robot_x, -1.5 - robot_y)  # Start at -1.5
    #         xy_end = (x - robot_x, 1.5 - robot_y)     # End at 1.5
    #         coor_start = self.to_im_coor(xy_start, res, self.m2pixel)
    #         coor_end = self.to_im_coor(xy_end, res, self.m2pixel)
    #         pygame.draw.line(surface, colors.COLOR_SLAM_GRID_BORDER, coor_start, coor_end, 5)

    #     for y in [-1.5, 1.5]:
    #         xy_start = (-1.5 - robot_x, y - robot_y)  # Start at -1.5
    #         xy_end = (1.5 - robot_x, y - robot_y)     # End at 1.5
    #         coor_start = self.to_im_coor(xy_start, res, self.m2pixel)
    #         coor_end = self.to_im_coor(xy_end, res, self.m2pixel)
    #         pygame.draw.line(surface, colors.COLOR_SLAM_GRID_BORDER, coor_start, coor_end, 5)

    #     if self.number_landmarks() > 0:
    #         for i in range(len(self.markers[0,:])):
    #             xy = (lms_xy[0, i], lms_xy[1, i])
    #             coor_ = self.to_im_coor(xy, res, self.m2pixel)
    #             try:
    #                 surface.blit(self.lm_pics[self.taglist[i]-1],
    #                 (coor_[0]-7, coor_[1]-7))
    #             except IndexError:
    #                 surface.blit(self.lm_pics[-1],
    #                 (coor_[0]-7, coor_[1]-7))
    
    #     #print(grocery_idx, shopping_list)
    #     if len(fruit_pose_dict) > 0:
    #         for key in fruit_pose_dict:
    #             robot_xy = self.robot.state[:2, 0]
    #             b_known = fruit_pose_dict[key]["known"]
    #             xy = (fruit_pose_dict[key]["x"] - robot_xy[0], fruit_pose_dict[key]["y"] - robot_xy[1])
    #             coor_ = self.to_im_coor(xy, res, self.m2pixel)
    #             fruit_name = key.split("_")[0]
    #             if b_known:

    #                 if is_shopping_fruit(fruit_name, shopping_list):
    #                     fruit_circle_colour = colors.COLOR_VISITED_FRUIT
    #                     if grocery_idx < len(shopping_list):
    #                         # check if fruit is finished
    #                         fruit_name = key.split('_')[0]
    #                         shopping_name_order = [shopping_list[i][0] for i in range(len(shopping_list))]
    #                         order = shopping_name_order.index(fruit_name)
    #                         if order > grocery_idx: 
    #                             fruit_circle_colour = colors.COLOR_NOT_VISITED_FRUIT
    #                         goal_fruit_id = shopping_list[grocery_idx][0]
    #                         goal_fruit_pos = shopping_list[grocery_idx][1]
    #                         fruit_pos = (fruit_pose_dict[key]["x"], fruit_pose_dict[key]["y"])
    #                         if (goal_fruit_id == fruit_name):
    #                             fruit_circle_colour = colors.COLOR_GOAL_FRUIT
    #                     pygame.draw.circle(surface, fruit_circle_colour, (coor_[0], coor_[1]), radius=15)

    #             surface.blit(self.fruit_pics[fruit_name],
    #                 (coor_[0]-8, coor_[1]-8))
                
    #     # Draw waypoints and lines between them
    #     if wp_manager:
    #         if len(wp_manager.wps) > 0:

    #             previous_coor = None
    #             for i in range(len(wp_manager.wps)):
    #                 w = wp_manager.wps[i]
    #                 robot_xy = self.robot.state[:2, 0]
    #                 coor_ = self.to_im_coor([w[0]-robot_xy[0], w[1]-robot_xy[1]], res, self.m2pixel)
                    
    #                 if i < wp_manager.current_wp:
    #                     colour = colors.COLOR_FINISHED_WP
    #                 elif i == wp_manager.current_wp:
    #                     colour = colors.COLOR_NEXT_WP
    #                 else:
    #                     colour = colors.COLOR_WP
    #                 radius = 4
    #                 pygame.draw.circle(surface, colour, (coor_[0], coor_[1]), radius)
    #                 # Draw line between current and previous waypoint
    #                 if previous_coor is not None:
    #                     pygame.draw.line(surface, colour, previous_coor, (coor_[0], coor_[1]), 2) # green line
    #                 previous_coor = (coor_[0], coor_[1])

        
    #     # draw robot on surface
    #     pibot_pic = pygame.image.load(colors.pi_bot_img)
    #     surface.blit(self.rot_center(pibot_pic, robot_theta*57.3),
    #                 (start_point_uv[0]-23, start_point_uv[1]-23))
        
    #     return surface



    ##########################################
    ##########################################
    ##########################################

    @staticmethod
    def umeyama(from_points, to_points):

    
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"
        
        N = from_points.shape[1]
        m = 2
        
        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        
        cov_matrix = delta_to @ delta_from.T / N
        
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        
        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
    
        return R, t

    @ staticmethod
    # every coordinate is between -1.5 to 1.5m (3 x 3 arena)
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)
    
    @ staticmethod
    def to_xy_coor(xy_im, res, m2pixel):
        w, h = res
        x_im, y_im = xy_im
        x = float((x_im-w/2.0)/-m2pixel)
        y = float((y_im-h/2.0)/m2pixel)
        return (x, y)

    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image       

    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        if abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle