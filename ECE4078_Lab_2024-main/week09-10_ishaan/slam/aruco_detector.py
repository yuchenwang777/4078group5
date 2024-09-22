# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:
    def __init__(self, robot, marker_length=0.07):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters() # updated to work with newer OpenCV
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) # updated to work with newer OpenCV
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            if idi in seen_ids or idi>10:
                continue
            else:
                seen_ids.append(idi)
            lm_tvecs = tvecs[ids==idi].T
            lm_rvecs = rvecs[ids==idi].T
            r_box=0.04 #Assumes cube side length is 8cm
            z_vec=np.array([0,0,r_box])

            idx=0
            lm_net = np.copy(lm_tvecs.T)
            if lm_rvecs.size>0:
                for el in lm_rvecs.T:
                    rot_mat, jac = cv2.Rodrigues(el)
                    norm_vec = rot_mat @ z_vec
                    #print(norm_vec)
                    lm_net[idx]-=norm_vec
                    lm_net[idx,2]+=0.035 #camera offset
                    idx+=1

            lm_net=lm_net.T
            lm_bff2d = np.block([[lm_net[2,:]],[-lm_net[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)
            lm_measurement = measure.Marker(lm_bff2d, idi)
            measurements.append(lm_measurement)
  
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
