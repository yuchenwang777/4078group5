class colors():
        def __init__(self, mode=2):
            self.mode = mode
            self.num_modes = 4
            self.set_mode()
        
        def next_mode(self):
            self.mode += 1 
            if self.mode > self.num_modes - 1:
                 self.mode = 0
            self.set_mode()


        def set_mode(self):
            if self.mode == 0:
                self.COLOR_BG = (0, 0, 0)
                self.COLOR_SLAM_BG = (15, 15, 30)
                self.COLOR_SLAM_GRID = (0, 0, 255) 
                self.COLOR_SLAM_GRID_BORDER = (255, 0, 0) 
                self.COLOR_EKF_VIEW_BORDER = (0, 0, 0)

                self.COLOR_ROBOT_COVAR = (255, 255, 255)
                self.COLOR_LM_COVAR = (244, 69, 96)
                self.COLOR_VISITED_FRUIT = (50, 50, 50)
                self.COLOR_GOAL_FRUIT = (0, 255, 0)
                self.COLOR_NOT_VISITED_FRUIT = (200, 0, 0)
                self.COLOR_NEXT_WP = (0, 255, 0)
                self.COLOR_FINISHED_WP = (50, 50, 50)
                self.COLOR_WP = (255, 0, 0)
                self.COLOR_HEADING_TEXT  = (255, 255, 255)
                self.COLOR_TEXT = (255, 255, 255)
                self.COLOR_CAPTION_TEXT = (255, 255, 255)
                self.COLOR_PATH_BG = (0, 0, 0)
                self.COLOR_PATH_HAZARD_CIRCLE = (255, 0, 0)
                self.COLOR_PATH_BORDER = (100, 100, 100)
                self.COLOR_PATH_WPS = (0,255,0)
                self.COLOR_PATH_GRID = (0, 0, 0)
                self.COLOR_PATH_START = (0, 255, 0)
                self.COLOR_PATH_END = (255, 0, 0)
                self.COLOR_PATH_OPEN_NODE = (255, 255, 0)
                self.COLOR_PATH_CLOSE_NODE = (50, 50, 50)
                self.COLOR_COMMAND_BG = (0, 0, 0)

                self.pi_bot_img = f'pics/8bit/pibot_top1.png'
            
            elif self.mode == 1:
                # mode: girlypop
                self.COLOR_BG = (0, 0, 0)
                self.COLOR_SLAM_BG =  (250, 150, 226)
                self.COLOR_SLAM_GRID = (255, 255, 255)
                self.COLOR_SLAM_GRID_BORDER = (255, 255, 255)
                self.COLOR_EKF_VIEW_BORDER = (255, 255, 255)

                self.COLOR_ROBOT_COVAR = (255, 255, 255)
                self.COLOR_LM_COVAR = (244, 69, 96)
                self.COLOR_VISITED_FRUIT = (50, 50, 50)
                self.COLOR_GOAL_FRUIT = (0, 255, 0)
                self.COLOR_NOT_VISITED_FRUIT = (200, 0, 0)
                self.COLOR_NEXT_WP = (0, 255, 0)
                self.COLOR_FINISHED_WP = (50, 50, 50)
                self.COLOR_WP = (255, 0, 0)
                self.COLOR_HEADING_TEXT  = self.COLOR_SLAM_BG
                self.COLOR_TEXT = self.COLOR_SLAM_BG
                self.COLOR_COMMAND_BG = self.COLOR_BG

                self.COLOR_PATH_BG = (0, 0, 0)
                self.COLOR_PATH_HAZARD_CIRCLE = (135, 7, 240)
                self.COLOR_PATH_BORDER = self.COLOR_PATH_HAZARD_CIRCLE
                self.COLOR_PATH_WPS = (0, 255, 0)
                self.COLOR_PATH_GRID = (0, 0, 0)
                self.COLOR_PATH_START = (0, 255, 0)
                self.COLOR_PATH_END = (255, 0, 0)
                self.COLOR_PATH_OPEN_NODE = (0, 153, 255)
                self.COLOR_PATH_CLOSE_NODE = (255,255,255)

                self.pi_bot_img = f'pics/8bit/pibot_top2.png'
            
            elif self.mode == 2:
                # Tom's mode
                self.COLOR_BG = (0, 0, 0)

                self.COLOR_SLAM_BG = (0, 118, 8)
                self.COLOR_SLAM_GRID = (0, 71, 10)
                self.COLOR_SLAM_GRID_BORDER = (0, 0, 0) 
                self.COLOR_EKF_VIEW_BORDER = (0, 71, 10)

                self.COLOR_ROBOT_COVAR = (255, 255, 255)
                self.COLOR_LM_COVAR = (244, 69, 96)
                self.COLOR_VISITED_FRUIT = (50, 50, 50)
                self.COLOR_GOAL_FRUIT = (0, 255, 0)
                self.COLOR_NOT_VISITED_FRUIT = (200, 0, 0)
                self.COLOR_NEXT_WP = (0, 255, 0)
                self.COLOR_FINISHED_WP = (50, 50, 50)
                self.COLOR_WP = (255, 0, 0)

                self.COLOR_HEADING_TEXT  = (0, 255, 0)
                self.COLOR_TEXT = (0, 255, 0)
                self.COLOR_CAPTION_TEXT = (0, 255, 0)

                self.COLOR_PATH_BG = (100, 100, 100)
                self.COLOR_PATH_HAZARD_CIRCLE = (50, 50, 50)
                self.COLOR_PATH_BORDER = (0, 0, 0)
                self.COLOR_PATH_WPS = (0,255,0)
                self.COLOR_PATH_GRID = (66, 16, 101)
                self.COLOR_PATH_START = (0, 255, 0)
                self.COLOR_PATH_END = (255, 0, 0)
                self.COLOR_PATH_OPEN_NODE = (255, 0, 255)
                self.COLOR_PATH_CLOSE_NODE = (0, 0, 0)
                self.COLOR_COMMAND_BG = (0, 43, 6)

                self.pi_bot_img = f'pics/8bit/pibot_top3.png'

            elif self.mode == 3:
                # CRISIS MODE
                self.COLOR_BG = (255, 0, 0)

                self.COLOR_SLAM_BG = (118, 8, 0)
                self.COLOR_SLAM_GRID = (71, 10, 0)
                self.COLOR_SLAM_GRID_BORDER = (0, 0, 0) 
                self.COLOR_EKF_VIEW_BORDER = (71, 0, 10)

                self.COLOR_ROBOT_COVAR = (255, 255, 255)
                self.COLOR_LM_COVAR = (255, 255, 255)
                self.COLOR_VISITED_FRUIT = (50, 50, 50)
                self.COLOR_GOAL_FRUIT = (0, 255, 0)
                self.COLOR_NOT_VISITED_FRUIT = (0, 0, 0)
                self.COLOR_NEXT_WP = (0, 255, 0)
                self.COLOR_FINISHED_WP = (50, 50, 50)
                self.COLOR_WP = (255, 255, 255)

                self.COLOR_HEADING_TEXT  = (100, 0, 0)
                self.COLOR_TEXT = (255, 0, 0)
                self.COLOR_CAPTION_TEXT = (255, 0, 0)

                self.COLOR_PATH_BG = (255, 100, 100)
                self.COLOR_PATH_HAZARD_CIRCLE = (50, 50, 50)
                self.COLOR_PATH_BORDER = (0, 0, 0)
                self.COLOR_PATH_WPS = (0,255,0)
                self.COLOR_PATH_GRID = (66, 16, 101)
                self.COLOR_PATH_START = (0, 255, 0)
                self.COLOR_PATH_END = (255, 0, 0)
                self.COLOR_PATH_OPEN_NODE = (255, 0, 255)
                self.COLOR_PATH_CLOSE_NODE = (0, 0, 0)
                self.COLOR_COMMAND_BG = (43, 0, 6)

                self.pi_bot_img = f'pics/8bit/pibot_top4.png'