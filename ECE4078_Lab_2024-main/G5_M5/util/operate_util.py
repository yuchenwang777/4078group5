## Utility functions used in operate.py
import json
import numpy as np
import matplotlib.pyplot as plt
import re


def read_search_list(fname):
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open(fname, 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list

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

def read_lab_output(path_to_slam, path_to_target):
    # reading the data from the file M3 ReRun\lab_output\slam.txt
    with open(path_to_slam) as f:
        data = f.read()
           
    # reconstructing the data as a dictionary
    myDict = json.loads(data)

    # Extracting taglist x and y as 3 arrays
    taglist = myDict['taglist']
    map = myDict['map']
    x = map[0]
    y = map[1]
    # Sorting taglist in increasing order
    sorted_indices = np.argsort(taglist)
    sorted_taglist = np.sort(taglist)

    # Use the sorted_indices to sort x and y in increasing order
    sorted_x = [x[i] for i in sorted_indices]
    sorted_y = [y[i] for i in sorted_indices]

    # Storing back into a dictionary
    myDict_aruco = {f"aruco{i+1}_0": {"y": sorted_y[i], "x": sorted_x[i]} for i in range(10)}
    # Retreiving the targets
    with open(path_to_target) as f:
        data = f.read()
    # reconstructing the data as a dictionary
    myDict_targets = json.loads(data)

    # Merging the targests and the aruco dicts
    myDict_aruco.update(myDict_targets)
    M3_generated_GT = myDict_aruco
    # Convert the dictionary to a JSON-formatted string
    json_str = json.dumps(M3_generated_GT)
    # Specify the file path where you want to save the JSON data
    file_path = "est_gt.txt"
    with open(file_path, "w") as file:
        file.write(json_str)

    print(f"Dictionary saved to {file_path}")

    gt_dict = M3_generated_GT
    
    fruit_list = []
    fruit_true_pos = []
    aruco_true_pos = np.empty([10, 2])

    # remove unique id of targets of the same type
    for key in gt_dict:
        x = np.round(gt_dict[key]['x'], 2)
        y = np.round(gt_dict[key]['y'], 2)

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


def plot_route(shopping_list_file,GT_file,routes=None):
    shopping_list = []
    with open(shopping_list_file, 'r') as fd:
        fruits = fd.readlines()
        for fruit in fruits:
            shopping_list.append(fruit.strip())
    
    with open(GT_file, 'r') as fd:
        GT = json.load(fd)

    class_colour = {
        'orange': np.array([255, 200, 0])/255,
        'lemon': np.array([200, 200, 150])/255,
        'lime': np.array([0, 255, 0])/255,
        'tomato': np.array([255, 0, 0])/255,
        'capsicum': np.array([50, 50, 251])/255,
        'potato': np.array([143,95, 0])/255,
        'pumpkin': np.array([255, 117, 24])/255,
        'garlic': np.array([211, 211, 211])/255
    }
    fig,ax = plt.subplots(1,1)

    if GT is not None:
        if isinstance(GT,dict):
            for key in GT.keys():
                if 'aruco' in key:
                    ax.scatter(GT[key]['x'],GT[key]['y'],label=key,marker='s',c='black')
                    ax.annotate(re.findall('\d+',key)[0],(GT[key]['x'],GT[key]['y']+0.1))
                else:
                    target = key.split('_')[0]

                    if target in shopping_list:
                        ax.scatter(GT[key]["x"],GT[key]["y"],label=key,color=class_colour[key.split('_')[0]],edgecolors='black')
                        fruit = plt.Circle((GT[key]["x"],GT[key]["y"]),radius=0.5,fill=True,alpha=0.5,color=class_colour[target])
                        ax.annotate(f"{target}_{shopping_list.index(target)}",(GT[key]["x"],GT[key]["y"]),color='blue')
                        ax.add_artist(fruit)                
                    else:
                        ax.scatter(GT[key]["x"],GT[key]["y"],label=key,color=class_colour[key.split('_')[0]],edgecolors='black')
                        ax.annotate(target,(GT[key]["x"],GT[key]["y"]),color='black')

        elif isinstance(GT,np.ndarray):
            txt = [1, 2, 6, 10, 5, 7, 4, 9, 8, 3]
            for i, tag in enumerate(txt):
                ax.annotate(tag, (GT[0,i], GT[1,i]))

    if routes is not None:

        ax.arrow(0,0,routes[0][0][0],routes[0][0][1],width=0.001,head_width=0.04,length_includes_head=True,facecolor=class_colour[shopping_list[0]],edgecolor='black')
        for i,route in enumerate(routes):
            ax.scatter(route[-1][0],route[-1][1],label=shopping_list[i],marker='o',color=class_colour[shopping_list[i]])
            for ind in range(len(route)-1):
                ax.arrow(route[ind][0],route[ind][1],route[ind+1][0]-route[ind][0],route[ind+1][1]-route[ind][1],width=0.001,head_width=0.04,length_includes_head=True,facecolor=class_colour[shopping_list[i]],edgecolor='black')
        for i in range(len(routes)-1):
            ax.arrow(routes[i][-1][0],routes[i][-1][1],routes[i+1][0][0]-routes[i][-1][0],routes[i+1][0][1]-routes[i][-1][1],width=0.001,head_width=0.04,length_includes_head=True,facecolor=class_colour[shopping_list[i+1]],edgecolor='black')
    
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], xlabel='x', ylabel='y')
    ax.set_aspect('equal')
    plt.show()

def ang_clamp(ang,min_value=-np.pi,max_value=np.pi):
    return (ang + max_value) % (max_value-min_value) + min_value

classLabels = {'0': 'tomato',
                       '1': 'capsicum',
                        '2': 'garlic',
                        '3': 'lemon',
                        '4': 'lime',
                        '5': 'orange',
                        '6': 'potato',
                        '7': 'pumpkin',
                        'tomato': 'tomato',
                        'capsicum': 'capsicum',
                        'garlic': 'garlic',
                        'lemon': 'lemon',
                        'lime': 'lime',
                        'orange': 'orange',
                        'potato': 'potato',
                        'pumpkin': 'pumpkin'}
def estimate_object_pose(camera_matrix, obj_info):
    """
    function:
        estimate the pose of a target based on size and location of its bounding box and the corresponding robot pose
    input:
        camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
            |f_x, s,   c_x|
            |0,   f_y, c_y|
            |0,   0,   1  |
            (f_x, f_y): focal length in pixels
            (c_x, c_y): optical centre in pixels
            s: skew coefficient (should be 0 for PenguinPi)
        obj_info: list, an individual bounding box in an image (generated by get_bounding_box, [label,[x,y,width,height]])
        robot_pose: list, pose of robot corresponding to the image (read from 'lab_output/images.txt', [x,y,theta])
    output:
        target_pose: dict, prediction of relative pose
    """
    # read in camera matrix (from camera calibration results)
    focal_length = camera_matrix[0][0]

    # there are 8 possible types of fruits and vegs
    ######### Replace with your codes #########
    # TODO: measure actual sizes of targets [width, depth, height] and update the dictionary of true target dimensions
    target_dimensions_dict = {'orange': [0.078,0.075,0.073], 'lemon': [0.074,0.05,0.052], 
                              'lime': [0.07,0.05,0.051], 'tomato': [0.07,0.06,0.06], 
                              'capsicum': [0.073,0.073,0.09], 'potato': [0.09,0.069,0.058], 
                              'pumpkin': [0.08,0.08,0.07], 'garlic': [0.065,0.06,0.076]} 
    #########

    # estimate target pose using bounding box and robot pose
    target_class = classLabels[obj_info[0].lower()]     # get predicted target label of the box
    target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
    true_height = target_dimensions_dict[target_class][2]   # look up true height of by class label

    # compute pose of the target based on bounding box info, true object height, and robot's pose
    pixel_height = target_box[3]
    pixel_center = target_box[0]
    distance = true_height/pixel_height * focal_length  # estimated distance between the robot and the centre of the image plane based on height
    # training image size 320x240p
    image_width = 320 # change this if your training image is in a different size (check details of pred_0.png taken by your robot)
    x_shift = image_width/2 - pixel_center              # x distance between bounding box centre and centreline in camera view
    theta = np.arctan(x_shift/focal_length)     # angle of object relative to the robot
    
   # relative object location
    distance_obj = distance/np.cos(theta) # relative distance between robot and object
    x_relative = distance_obj * np.cos(theta) # relative x pose
    y_relative = distance_obj * np.sin(theta) # relative y pose
    relative_pose = {'x': x_relative, 'y': y_relative}
    #print(f'relative_pose: {relative_pose}')
    return relative_pose
    # location of object in the world frame using rotation matrix
    delta_x_world = x_relative * np.cos(robot_pose[2]) - y_relative * np.sin(robot_pose[2])
    delta_y_world = x_relative * np.sin(robot_pose[2]) + y_relative * np.cos(robot_pose[2])
    # add robot pose with delta target pose
    target_pose = {'y': (robot_pose[1]+delta_y_world)[0],
                   'x': (robot_pose[0]+delta_x_world)[0]}
    #print(f'delta_x_world: {delta_x_world}, delta_y_world: {delta_y_world}')
    #print(f'target_pose: {target_pose}')

    # return target_pose