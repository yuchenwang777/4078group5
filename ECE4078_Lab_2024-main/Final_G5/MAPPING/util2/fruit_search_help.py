import numpy as np
import json
import os

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
    

def read_search_list(shopping_list_name):
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    file_path = os.path.join(os.path.dirname(__file__), f'../{shopping_list_name}')
    with open(file_path, 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def order_shopping_list(fruit_order, unordered_shopping_list):
    ordered_shopping_list = []
    
    for fruit in fruit_order:
        for item in unordered_shopping_list:
            if item[0] == fruit:
                ordered_shopping_list.append(item)
                break
                
    return ordered_shopping_list

def is_shopping_fruit(fruit_name, shopping_list):
    for fruit in shopping_list:
        if fruit[0] == fruit_name:
            return True
    return False