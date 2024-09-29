
def GUI(self, markers_locations, fruit_true_pos, fruit_list, res=(320, 500), not_pause=True):
    
    fruit_colors = {
        'orange': (0, 165, 255),
        'lemon': (0, 255, 255),
        'lime': (0, 255, 0),
        'tomato': (0, 0, 255),
        'capsicum': (255, 0, 0),
        'potato': (255, 255, 0),
        'pumpkin': (255, 165, 0),
        'garlic': (255, 0, 255)
    }

    # Draw landmarks
    m2pixel = 100
    if not_pause:
        bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
    else:
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
    canvas = np.ones((res[1], res[0], 3)) * bg_rgb.astype(np.uint8)

    robot_xy = self.robot.state[:2, 0].reshape((2, 1))
    robot_xy = robot_xy * 0
    robot_theta = self.robot.state[2,0]

    # Plot robot
    start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
    p_robot = self.P[0:2, 0:2]
    axes_len, angle = self.make_ellipse(p_robot)

    canvas = cv2.ellipse(canvas, start_point_uv,
                        (int(axes_len[0] * m2pixel), int(axes_len[1] * m2pixel)),
                        angle, 0, 360, (0, 30, 56), 1)

    # Draw landmarks
    if self.number_landmarks() > 0:
        for i in range(len(markers_locations[0, :])):
            xy = (markers_locations[0, i], markers_locations[1, i])
            coor_ = self.to_im_coor(xy, res, m2pixel)

            # Plot covariance
            Plmi = self.P[3 + 2 * i:3 + 2 * (i + 1), 3 + 2 * i:3 + 2 * (i + 1)]
            axes_len, angle = self.make_ellipse(Plmi)

            canvas = cv2.ellipse(canvas, coor_,
                                (int(axes_len[0] * m2pixel), int(axes_len[1] * m2pixel)),
                                angle, 0, 360, (244, 69, 96), 1)

    # Draw fruits 
    for fruit_pos, fruit_name in fruit_true_pos:
        if fruit_name in fruit_colors:
            fruit_color = fruit_colors[fruit_name]

        fruit_xy = (fruit_pos[0], fruit_pos[1])
        fruit_coor = self.to_im_coor(fruit_xy, res, m2pixel)
        cv2.circle(canvas, fruit_coor, 5, fruit_color, -1) 

    # Draw list
    for fruit_name, fruit_pos in fruit_list:
        print(fruit_name,fruit_pos)
        if fruit_name in fruit_colors:
            fruit_color = fruit_colors[fruit_name]
        else:
            fruit_color = (255, 255, 255)  

        fruit_xy = (fruit_pos[0], fruit_pos[1])
        fruit_coor = self.to_im_coor(fruit_xy, res, m2pixel)
        
        
        star_size = 5  
        line_length = 10 
        cv2.line(canvas, (fruit_coor[0] - star_size, fruit_coor[1]), (fruit_coor[0] + star_size, fruit_coor[1]), fruit_color, 2)
        cv2.line(canvas, (fruit_coor[0], fruit_coor[1] - star_size), (fruit_coor[0], fruit_coor[1] + star_size), fruit_color, 2)
        cv2.line(canvas, (fruit_coor[0] - line_length, fruit_coor[1] - line_length), (fruit_coor[0] + line_length, fruit_coor[1] + line_length), fruit_color, 2)
        cv2.line(canvas, (fruit_coor[0] - line_length, fruit_coor[1] + line_length), (fruit_coor[0] + line_length, fruit_coor[1] - line_length), fruit_color, 2)

    surface = pygame.surfarray.make_surface(np.rot90(canvas))
    surface = pygame.transform.flip(surface, True, False)
    surface.blit(self.rot_center(self.pibot_pic, robot_theta * 57.3),
                (start_point_uv[0] - 15, start_point_uv[1] - 15))

    if self.number_landmarks() > 0:
        for i in range(len(markers_locations[0, :])):
            xy = (markers_locations[0, i], markers_locations[1, i])
            coor_ = self.to_im_coor(xy, res, m2pixel)

            try:
                surface.blit(self.lm_pics[self.taglist[i] - 1],
                            (coor_[0] - 5, coor_[1] - 5))
            except IndexError:
                surface.blit(self.lm_pics[-1],
                            (coor_[0] - 5, coor_[1] - 5))

    return surface
