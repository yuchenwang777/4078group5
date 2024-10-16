import numpy as np
from copy import deepcopy

def match_aruco_points(aruco0: dict, aruco1: dict):
    points0 = []
    points1 = []
    keys = []
    for key in aruco0:
        if not key in aruco1:
            continue

        points0.append(aruco0[key])
        points1.append(aruco1[key])
        keys.append(key)
        
    return keys, np.hstack(points0), np.hstack(points1)

def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])

    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1 / num_points * np.reshape(np.sum(points1, axis=1), (2, -1))
    mu2 = 1 / num_points * np.reshape(np.sum(points2, axis=1), (2, -1))
    sig1sq = 1 / num_points * np.sum((points1 - mu1) ** 2.0)
    sig2sq = 1 / num_points * np.sum((points2 - mu2) ** 2.0)
    Sig12 = 1 / num_points * (points2 - mu2) @ (points1 - mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1, -1] = -1

    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1, 0], R[0, 0])
    x = mu2 - R @ mu1

    return theta, x

def apply_transform(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert (points.shape[0] == 2)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    points_transformed = R @ points + x
    
    return points_transformed

def compute_slam_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1 - points2).ravel()
    MSE = 1.0 / num_points * np.sum(residual ** 2)

    return np.sqrt(MSE)

def align_object_poses(theta, x, objects_est):
    objects = deepcopy(objects_est)

    for object in objects:
        poses = []
        for pos in objects[object]:
            pos = np.reshape(pos, (2, 1))
            pos = apply_transform(theta, x, pos)
            pos = np.reshape(pos, (1, 2))[0]

            poses.append(pos)

        objects[object] = poses

    return objects

def compute_object_est_error(gt_list, est_list):
    """Compute the object target pose estimation error based on Euclidean distance

    If there are more estimations than the number of targets (e.g. only 1 target orange, but detected 2),
        then take the average error of the 2 detections

    if there are fewer estimations than the number of targets (e.g. 2 target oranges, but only detected 1),
        then return [MAX_ERROR, error with the closest target]

    @param gt_list: target ground truth list
    @param est_list: estimation list
    @return: error of all the objects
    """

    MAX_ERROR = 1

    object_errors = {}

    for target_type in gt_list:
        n_gt = len(gt_list[target_type])  # number of targets in this fruit type

        type_errors = []
        for i, gt in enumerate(gt_list[target_type]):
            dist = []
            try:
                for est in est_list[target_type]:
                    dist.append(np.linalg.norm(gt - est))  # compute Euclidean distance
    
                n_est = len(est_list[target_type])
    
                # if this fruit type has been detected
                if len(dist) > 0:
                    if n_est > n_gt:    # if more estimation than target, take the mean error
                        object_errors[target_type + '_{}'.format(i)] = np.round(np.mean(dist), 3)
                    elif n_est < n_gt:  # see below
                        type_errors.append(np.min(dist))
                    else:   # for normal cases, n_est == n_gt, take the min error
                        object_errors[target_type + '_{}'.format(i)] = np.round(np.min(dist), 3)
            except:   # if there is no estimation for this fruit type
                for j in range(n_gt):
                    object_errors[target_type + '_{}'.format(j)] = MAX_ERROR

        if len(type_errors) > 0:    # for the n_est < n_gt scenario
            type_errors = np.sort(type_errors)
            for i in range(len(type_errors) - 1):
                object_errors[target_type + '_{}'.format(i+1)] = np.round(type_errors[i], 3)
            object_errors[target_type + '_{}'.format(0)] = MAX_ERROR



    return object_errors