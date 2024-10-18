import os
import random
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arg_parser import parse_args
args = parse_args()

# Class names as defined in your dataset generation script
class_names = ['capsicum', 'garlic', 'lemon', 'lime', 'pear', 'potato', 'pumpkin', 'tomato']

# Color mapping for each fruit class
class_colors = {
    'capsicum': (0, 255, 255),  # Yellow
    'garlic': (255, 255, 255),  # White
    'lemon': (0, 255, 255),     # Yellow
    'lime': (0, 255, 0),        # Green
    'pear': (102, 255, 102),    # Light Green
    'potato': (153, 102, 51),   # Brown
    'pumpkin': (0, 140, 255),   # Orange
    'tomato': (0, 0, 255)       # Red
}

# Function to load and draw bounding box on an image
def draw_bounding_box(img_path, label_path):
    # Load the image
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Load the bounding box label
    with open(label_path, 'r') as label_file:
        for line in label_file:
            label_data = line.strip().split()
            
            class_id = int(label_data[0])
            class_name = class_names[class_id]  # Get the class name from the class ID
            norm_bbox = list(map(float, label_data[1:]))
            print(f"Class: {class_name}, Normalized Bounding Box: {norm_bbox}")

            # Denormalize the bounding box
            x1 = int((norm_bbox[0] * width) - ((norm_bbox[2] * width)/2))
            y1 = int((norm_bbox[1] * height) - ((norm_bbox[3] * height)/2))
            x2 = int((norm_bbox[0] * width) + ((norm_bbox[2] * width)/2))
            y2 = int((norm_bbox[1] * height) + ((norm_bbox[3] * height)/2))

            # Get the color associated with the class name
            color = class_colors.get(class_name, (0, 255, 0))  # Default to green if no match

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Calculate the position for the label
            label_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_x = x1
            label_y = y1 - 10
            
            # Adjust label position if it goes out of the image
            if label_y < 0:  # If the label is above the image, draw it inside the box
                label_y = y1 + label_size[1] + 10

            if label_x + label_size[0] > width:  # If the label is too wide, shift it left
                label_x = width - label_size[0] - 10

            # Display the class name on the bounding box
            cv2.putText(img, class_name, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def main():



    dataset_to_examine = 'dataset_'+ str(args.dataset_id)



    # Determine the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Base directory relative to the script's location
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'yolo', 'datasets', dataset_to_examine))

    # Sub-directories for train, test, val
    sets = ['train', 'val', 'test']

    while True:
        # Select a random set (train, val, test)
        chosen_set = random.choice(sets)

        # Get a random image from the chosen set
        images_dir = os.path.join(base_dir, chosen_set, 'images')
        labels_dir = os.path.join(base_dir, chosen_set, 'labels')

        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if not image_files:
            print(f"No images found in {images_dir}")
            return

        random_image = random.choice(image_files)
        img_path = os.path.join(images_dir, random_image)
        label_path = os.path.join(labels_dir, random_image.replace('.jpg', '.txt'))

        # Draw bounding box on the image
        img_with_bbox = draw_bounding_box(img_path, label_path)

        # Display the image with bounding box
        cv2.imshow('Random Image with Bounding Box', img_with_bbox)
        
        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Exit if 'q' is pressed
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()