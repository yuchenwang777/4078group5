import os
import cv2
import numpy as np
import sys
import random
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arg_parser import parse_args

# Parse the number of flipped images
args = parse_args()
num_flipped_images = int(args.num_flipped_images)  # Convert to integer

script_dir = os.path.dirname(os.path.abspath(__file__))
original_dataset_path = os.path.abspath(os.path.join(script_dir, '..', 'datasets', f'dataset_{args.dataset_id}'))

# Generate unique dataset path
def generate_unique_dataset_path(base_path):
    # Start with the base path
    unique_path = base_path
    # Check if the path exists, if so, append a number to make it unique
    i = 1
    while os.path.exists(unique_path):
        unique_path = f"{base_path}_{i}"
        i += 1
    return unique_path

base_new_dataset_path = os.path.abspath(os.path.join(script_dir, '..', 'datasets', f'dataset_{args.dataset_id}_flipped'))
new_dataset_path = generate_unique_dataset_path(base_new_dataset_path)

print(f"Original dataset path: {original_dataset_path}")
print(f"New dataset path: {new_dataset_path}")

splits = ['train', 'val', 'test']
# Create flipped dataset directories (train, val, test, images, labels)
os.makedirs(new_dataset_path, exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(new_dataset_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, split, 'labels'), exist_ok=True)


# Helper function to flip bounding boxes
def flip_bounding_box(bbox, img_width):
    class_id, x_center, y_center, width, height = bbox
    # Flip horizontally: (1 - x_center) in normalized coordinates
    x_center_flipped = 1 - x_center
    return [class_id, x_center_flipped, y_center, width, height]

# Collect all image files across all splits
all_image_files = []
for split in splits:
    images_dir = os.path.join(original_dataset_path, split, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    all_image_files.extend([(split, img) for img in image_files])

# Check if the requested number of flipped images is valid
if num_flipped_images > len(all_image_files):
    raise ValueError(f"Requested number of flipped images ({num_flipped_images}) exceeds available images ({len(all_image_files)}) in the dataset.")

# Randomly select the number of images to flip
selected_image_files = random.sample(all_image_files, num_flipped_images)

# Loop through selected image files and flip them
for split, img_file in selected_image_files:
    images_dir = os.path.join(original_dataset_path, split, 'images')
    labels_dir = os.path.join(original_dataset_path, split, 'labels')

    # Image path
    img_path = os.path.join(images_dir, img_file)
    # Corresponding label file
    label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(labels_dir, label_file)

    # Read the image
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # Flip the image horizontally
    flipped_img = cv2.flip(img, 1)

    # Read the bounding box labels
    with open(label_path, 'r') as f:
        bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]

    # Ensure the class ID is an integer
    for i in range(len(bboxes)):
        bboxes[i][0] = int(bboxes[i][0])

    # Flip the bounding boxes
    flipped_bboxes = [flip_bounding_box(bbox, img_width) for bbox in bboxes]

    # Save the flipped image
    flipped_img_path = os.path.join(new_dataset_path, split, 'images', img_file)
    cv2.imwrite(flipped_img_path, flipped_img)

    # Save the new label file
    flipped_label_path = os.path.join(new_dataset_path, split, 'labels', label_file)
    with open(flipped_label_path, 'w') as f:
        for bbox in flipped_bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')

# Generate data.yaml file
class_names = ['capsicum', 'garlic', 'lemon', 'lime', 'pear', 'potato', 'pumpkin', 'tomato']
yaml_content = f"""
train: {os.path.join(new_dataset_path, 'train', 'images')}
val: {os.path.join(new_dataset_path, 'val', 'images')}
test: {os.path.join(new_dataset_path, 'test', 'images')}

nc: {len(class_names)}
names: {class_names}
"""

yaml_path = os.path.join(new_dataset_path, "data.yaml")
with open(yaml_path, 'w') as yaml_file:
    yaml_file.write(yaml_content)

print(f"Flipped dataset created successfully with {num_flipped_images} images!")
print(f"The new dataset is saved as: {new_dataset_path}")