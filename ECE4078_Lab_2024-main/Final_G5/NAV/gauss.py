import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def get_gaussian_value_map(obs, sig_obs, sig_border, grid_size, grid_res):
    value_map = np.zeros( (int(grid_size / grid_res), int(grid_size / grid_res)) )
    for i in range(-int(grid_size / (2*grid_res)), int(grid_size / (2*grid_res))):
        for j in range(-int(grid_size / (2*grid_res)), int(grid_size / (2*grid_res))):
            total_value = 0.0
            # Obs
            for ob in obs:
                ob_x = int((ob[0] + (grid_size/2))/(grid_res)) - (grid_size/(2*grid_res))
                ob_y = int((ob[1] + (grid_size/2))/(grid_res)) - (grid_size/(2*grid_res))
                distance_to_obs = math.sqrt((i - ob_x)**2 + (j - ob_y)**2)
                obst_gaus = math.exp(-distance_to_obs ** 2 / ( 2 * sig_obs ** 2) )
                total_value = max(total_value, obst_gaus)

            # border
            distance_to_border = min(abs(i+int(grid_size / (2*grid_res))), abs(j+int(grid_size / (2*grid_res))), abs(int(grid_size / (2*grid_res))-i), abs(int(grid_size / (2*grid_res))-j))
            border_gaus = math.exp(-distance_to_border ** 2 / ( 2 * sig_border ** 2) )
            total_value = max(total_value, border_gaus)
            value_map[i+int(grid_size / (2*grid_res)), j+int(grid_size / (2*grid_res))] = total_value

    #cv.imwrite("gaus_img.png", cv.rotate(cv.resize(255*value_map, (500, 500)), cv.ROTATE_90_CLOCKWISE))
    cv.imwrite("pics/gaus_img.png", cv.rotate(cv.resize(255*value_map, (500, 500)), cv.ROTATE_90_CLOCKWISE))
    return value_map