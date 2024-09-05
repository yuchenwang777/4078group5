import cv2
import os
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

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
            bboxes: list of lists, box info [label,[x,y,width,height]] for all detected targets in image
            img_out: image with bounding boxes and class labels drawn on
        """
        bboxes = self._get_bounding_boxes(img)

        img_out = deepcopy(img)

        # draw bounding boxes on the image
        for bbox in bboxes:
            #  translate bounding box info back to the format of [x1,y1,x2,y2]
            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            # draw bounding box
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), self.class_colour[bbox[0]], thickness=2)

            # draw class label
            img_out = cv2.putText(img_out, bbox[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  self.class_colour[bbox[0]], 2)

        return bboxes, img_out

    def _get_bounding_boxes(self, cv_img):
        """
        function:
            get bounding box and class label of target(s) in an image as detected by YOLOv8
        input:
            cv_img    : image, e.g., image read by the cv2.imread() function
            model_path: str, e.g., 'yolov8n.pt', trained YOLOv8 model
        output:
            bounding_boxes: list of lists, box info [label,[x,y,width,height]] for all detected targets in image
        """

        # predict target type and bounding box with your trained YOLO

        predictions = self.model.predict(cv_img, imgsz=320, verbose=False)

        # get bounding box and class label for target(s) detected
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                # bounding format in [x, y, width, height]
                box_cord = box.xywh[0]

                box_label = box.cls  # class label of the box

                bounding_boxes.append([prediction.names[int(box_label)], np.asarray(box_cord)])

        return bounding_boxes


# FOR TESTING ONLY
if __name__ == '__main__':
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize the YOLO detector with the model
    yolo = Detector(f'{script_dir}/model/yolov8_model.pt')

    # Directory containing the images
    images_dir = f'{script_dir}/test/'

    # Loop through all files in the images directory
    for filename in os.listdir(images_dir):
        # Check if the file is an image (you can add more extensions if needed)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Full path to the image
            img_path = os.path.join(images_dir, filename)
            
            # Read the image
            img = cv2.imread(img_path)
            
            # Perform detection on the image
            bboxes, img_out = yolo.detect_single_image(img)
            
            # Print the bounding boxes and the number of detections
            print(f'Detections for {filename}:')
            print(bboxes)
            print(f'Number of detections: {len(bboxes)}')
            
            # Display the image with detections
            cv2.imshow(f'yolo detect - {filename}', img_out)
            cv2.waitKey(0)  # Wait for a key press to move to the next image
            cv2.destroyAllWindows()  # Close the window before the next image

   

