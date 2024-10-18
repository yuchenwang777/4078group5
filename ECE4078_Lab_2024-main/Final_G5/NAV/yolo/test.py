import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arg_parser import parse_args
from detector import Detector
import cv2
from arg_parser import parse_args

def main():
    # Parse command line arguments
    args = parse_args()

    # Get the current script directory and model path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = f'{base_dir}/models/{args.model_id}.pt'
    image_dir = f'{base_dir}/test_imgs/'

    # Initialize YOLO detector
    yolo = Detector(model_path)

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    image_files = sorted(image_files)  # Sort to ensure consistent order

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error loading image {image_path}")
            continue

        # Detect objects in the image
        bboxes, img_out = yolo.detect_single_image(img)

        # Print the bounding boxes and their count
        print(f"Detections for {image_file}:")
        print(bboxes)
        print(f"Number of detections: {len(bboxes)}")

        # Show the image with detections
        cv2.imshow('YOLO Detection', img_out)

        # Wait for user input
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()