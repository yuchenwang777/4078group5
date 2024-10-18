import os
import random
from PIL import Image
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arg_parser import parse_args

args = parse_args()

# Seed for reproducibility
random.seed(2)

dataset_id = args.new_dataset_id

# Determine the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate base directory relative to the script location
base_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'yolo'))

dataset_dir = os.path.join(base_dir, 'datasets', f'dataset_{dataset_id}')
print(f"Dataset directory: {dataset_dir}")

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)

os.makedirs(dataset_dir, exist_ok=True)

# Sub-directories for train, test, val
for subdir in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dataset_dir, subdir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, subdir, 'labels'), exist_ok=True)

# Paths to fruit and background images
fruit_dir = os.path.join(base_dir, 'generate_random_data', 'fruits')
background_dir = os.path.join(base_dir, 'generate_random_data', 'backgrounds')

fruits = [f for f in os.listdir(fruit_dir) if f.endswith('.png')]
backgrounds = [b for b in os.listdir(background_dir) if b.endswith('.png')]

# Parameters for transformations and data generation
scale_range = (0.5, 1.5)
rotation_range = (-15, 15)
sets = ['train', 'val', 'test']
sets_distribution = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test
repeats_per_fruit = args.repeats_per_fruit  # Number of times each fruit image is used

class_names = ['capsicum', 'garlic', 'lemon', 'lime', 'pear', 'potato', 'pumpkin', 'tomato']

def get_label_index(fruit_name):
    for idx, name in enumerate(class_names):
        if fruit_name.startswith(name):
            return idx
    return None  # In case the fruit name does not match any class

def random_transform(image):
    scale = random.uniform(*scale_range)
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    rotation = random.uniform(*rotation_range)
    return image.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))

def composite_image(fruit_path, background_path, set_name, instance):
    fruit = Image.open(fruit_path).convert('RGBA')
    background = Image.open(background_path).convert('RGBA')
    fruit_transformed = random_transform(fruit)

    # Calculate effective dimensions after transformation
    fruit_mask = fruit_transformed.split()[-1]  # Get the alpha channel as a mask
    bbox = fruit_mask.getbbox()
    fruit_transformed = fruit_transformed.crop(bbox)
    fruit_width = bbox[2] - bbox[0]
    fruit_height = bbox[3] - bbox[1]

    # Ensure 100%of the fruit is visible
    
    x_min = 0
    x_max = background.width - fruit_width
    y_min = 0
    y_max = background.height - fruit_height
    if y_max < 0:
        y_max = 0

    #print(background.height,  fruit_height)
    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)
    #print(x, y)
    background.paste(fruit_transformed, (x, y), fruit_transformed)
    background_rgb = background.convert('RGB')

    file_name = f"{os.path.splitext(os.path.basename(fruit_path))[0]}_{instance}.jpg"
    img_path = os.path.join(dataset_dir, set_name, 'images', file_name)
    background_rgb.save(img_path)

    # Calculate the visible bounding box
    visible_x1 = max(x, 0)
    visible_y1 = max(y, 0)
    visible_x2 = min(x + fruit_width, background.width)
    visible_y2 = min(y + fruit_height, background.height)

    # Normalize bounding box
    norm_bbox = [
        ((visible_x1 + visible_x2) / 2) / background.width,  # center x
        ((visible_y1 + visible_y2) / 2) / background.height, # center y
        (visible_x2 - visible_x1) / background.width,      # width
        (visible_y2 - visible_y1) / background.height      # height
    ]

    label_index = get_label_index(os.path.basename(fruit_path))
    label_path = os.path.join(dataset_dir, set_name, 'labels', file_name.replace('.jpg', '.txt'))
    with open(label_path, 'w') as label_file:
        label_file.write(f"{label_index} {' '.join(map(str, norm_bbox))}\n")
total_images = len(fruits) * repeats_per_fruit
images_processed = 0

# Image generation process
for fruit in fruits:
    for i in range(repeats_per_fruit):
        images_processed += 1
        percent_complete = (images_processed / total_images) * 100
        chosen_set = random.choices(sets, weights=sets_distribution)[0]
        background = random.choice(backgrounds)
        composite_image(
            os.path.join(fruit_dir, fruit),
            os.path.join(background_dir, background),
            chosen_set,
            i
        )
    print(f"{percent_complete:.2f}% - Processed {fruit}")

print(f"Generated {images_processed} images in total.")

# Generate data.yaml file
yaml_content = f"""
train: {os.path.join(dataset_dir, 'train', 'images')}
val: {os.path.join(dataset_dir, 'val', 'images')}
test: {os.path.join(dataset_dir, 'test', 'images')}

nc: {len(class_names)}
names: {class_names}
"""

yaml_path = os.path.join(dataset_dir, "data.yaml")
with open(yaml_path, 'w') as yaml_file:
    yaml_file.write(yaml_content)

print(f"data.yaml file created at {yaml_path}")

