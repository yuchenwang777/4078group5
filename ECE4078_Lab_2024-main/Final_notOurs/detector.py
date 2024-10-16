import cv2
import os
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops
import torch

class Detector:
    def __init__(self, model_path, confidence_threshold=0.8):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold  # Add confidence threshold

        self.class_colour = {
            'pear': (51, 255, 51),
            'lemon': (0, 255, 255),
            'lime': (0, 102, 0),
            'tomato': (0, 0, 255),
            'capsicum': (255, 255, 0),
            'potato': (0, 51, 102),
            'pumpkin': (0, 127, 255),
            'garlic': (255, 0, 255)
        }

    def detect_single_image(self, img):
        """
        function:
            detect target(s) in an image
        input:
            img: image, e.g., image read by the cv2.imread() function
        output:
            bboxes: list of lists, box info [label, confidence, [x,y,width,height]] for all detected targets in image
            img_out: image with bounding boxes and class labels drawn on
        """
        bboxes = self._get_bounding_boxes(img)

        img_out = deepcopy(img)

        # draw bounding boxes on the image
        for bbox in bboxes:
            #  translate bounding box info back to the format of [x1,y1,x2,y2]
            xyxy = ops.xywh2xyxy(bbox[2])  # bbox[2] contains the [x, y, width, height]
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            # draw bounding box
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), self.class_colour[bbox[0]], thickness=2)

            # draw class label with confidence
            label_text = f"{bbox[0]}: {bbox[1]:.2f}"  # bbox[0] is the label, bbox[1] is the confidence
            img_out = cv2.putText(img_out, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  self.class_colour[bbox[0]], 2)

        return bboxes, img_out

    def _get_bounding_boxes(self, cv_img):
        """
        function:
            get bounding box, class label, and confidence of target(s) in an image as detected by YOLOv8
        input:
            cv_img    : image, e.g., image read by the cv2.imread() function
        output:
            bounding_boxes: list of lists, box info [label, confidence, [x,y,width,height]] for all detected targets
        """

        # predict target type and bounding box with your trained YOLO
        predictions = self.model.predict(cv_img, imgsz=320, verbose=False)

        # get bounding box, class label, and confidence for target(s) detected
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                confidence = box.conf.cpu().item()  # Get confidence score
                if confidence >= self.confidence_threshold:  # Only keep boxes above threshold
                    # bounding format in [x, y, width, height]
                    box_cord = box.xywh[0].cpu().numpy()  # Get bounding box coordinates
                    box_label = box.cls  # class label of the box

                    bounding_boxes.append([prediction.names[int(box_label)], confidence, np.asarray(box_cord)])

        return bounding_boxes