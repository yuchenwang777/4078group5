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
            if idi in seen_ids or idi>10:
                continue
            else:
                seen_ids.append(idi)

             # Get the rotation and translation vectors
            rvec = rvecs[i]
            tvec = tvecs[i]

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Calculate the offset from the marker to the cube's center
            offset = np.array([0, 0, self.marker_length / 2]).reshape((3, 1))

            # Transform the marker position to the cube's center
            cube_center = tvec.T + R @ offset

            # Convert to 2D coordinates (assuming the robot's pose is aligned with the principal axis)
            lm_bff2d = np.block([[cube_center[2, 0]], [-cube_center[0, 0]]]).reshape(-1, 1)

            lm_measurement = measure.Marker(lm_bff2d, idi)
            measurements.append(lm_measurement)
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
