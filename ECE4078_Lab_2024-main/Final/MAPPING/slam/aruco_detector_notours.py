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
        # rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params) # use this instead if you got a value error

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            if idi in seen_ids or idi > 10:
                continue
            else:
                seen_ids.append(idi)
            cube = self.marker_length
            lm_tvecs = tvecs[ids==idi].T
            lm_rvecs = rvecs[ids==idi].T
            for i in range(lm_tvecs.shape[1]):
                rmat,_ = cv2.Rodrigues(lm_rvecs[:,i])
                tvec = lm_tvecs[:,i]

                tvec[2] = (tvec[2] + 0.0416824)/0.986147# Z y = 0.986147 x - 0.0416824
                tvec[0] = (tvec[0] + 0.0259)/1.0608#X = 1.0608x + 0.0259
                
                # shifting the point to the center by 3.5 cm 
                tvec = tvec + rmat[:, 2] * (0.04)

                lm_tvecs_col = tvec - cube / 2  * rmat[:,2]
                lm_tvecs[:,i] = lm_tvecs_col

            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)
            

            # Linearly scale covariance with distance
            dist = np.linalg.norm(lm_bff2d)
            if dist > 3*1.42:
                lm_cov = 99999999999
            lm_cov = 0.2*np.eye(2)*dist
            lm_measurement = measure.Marker(lm_bff2d, idi, lm_cov)  
            measurements.append(lm_measurement)
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
