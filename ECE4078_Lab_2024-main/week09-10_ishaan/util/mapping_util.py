""" 
Utility functions for RRTC in operate.py
"""
from path_planning.Obstacle import *



def create_obstacle_list(obstacle_xys,safe_radius):
    obstacleList = []
    for xy in obstacle_xys:
        obstacleList.append(Circle(xy[0],xy[1],radius=safe_radius))
    return obstacleList

def length_of_path(path):
    dist = 0.0
    for node,nodeNext in zip(path[:-1],path[1:]):
        dist += np.hypot(node[0]-nodeNext[0],node[1]-nodeNext[1])
    return dist