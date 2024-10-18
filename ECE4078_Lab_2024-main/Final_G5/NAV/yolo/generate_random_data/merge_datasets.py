import os
import shutil
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arg_parser import parse_args

def combine_datasets(args, dataset1_path, dataset2_path, combined_dataset_path):
    # Define paths for images and labels
    subdirs = ['train', 'val', 'test']
    image_dirs = {subdir: os.path.join(combined_dataset_path, subdir, 'images') for subdir in subdirs}
    label_dirs = {subdir: os.path.join(combined_dataset_path, subdir, 'labels') for subdir in subdirs}
    
    # Create directories for the combined dataset
    for subdir in subdirs:
        os.makedirs(image_dirs[subdir], exist_ok=True)
        os.makedirs(label_dirs[subdir], exist_ok=True)
    
    def copy_files(src_images, src_labels, dst_images, dst_labels, prefix=""):
        for subdir in subdirs:
            src_subdir_images = os.path.join(src_images, subdir, 'images')
            src_subdir_labels = os.path.join(src_labels, subdir, 'labels')
            dst_subdir_images = image_dirs[subdir]
            dst_subdir_labels = label_dirs[subdir]

            # Format paths for printing
            relative_src_images = os.path.relpath(src_subdir_images, start=dataset1_path)
            relative_src_labels = os.path.relpath(src_subdir_labels, start=dataset1_path)
            relative_dst_images = os.path.relpath(dst_subdir_images, start=dataset1_path)
            relative_dst_labels = os.path.relpath(dst_subdir_labels, start=dataset1_path)

            print(f"Copying from {relative_src_images} to {relative_dst_images}")
            print(f"Copying from {relative_src_labels} to {relative_dst_labels}")

            if not os.path.exists(src_subdir_images):
                print(f"Source images subdirectory does not exist: {src_subdir_images}")
                continue

            if not os.path.exists(src_subdir_labels):
                print(f"Source labels subdirectory does not exist: {src_subdir_labels}")
                continue

            image_files = [f for f in os.listdir(src_subdir_images) if f.endswith(('.jpg', '.png'))]
            label_files = [f for f in os.listdir(src_subdir_labels) if f.endswith('.txt')]

            total_files = len(image_files) + len(label_files)
            if total_files == 0:
                print(f"No files to copy in {subdir}")
                continue

            copied_files = 0
            with tqdm(total=total_files, desc=f"Processing {subdir}", unit="file") as pbar:
                for image_file in image_files:
                    src_image_path = os.path.join(src_subdir_images, image_file)
                    dst_image_path = os.path.join(dst_subdir_images, prefix + image_file)
                    shutil.copy(src_image_path, dst_image_path)
                    pbar.update(1)

                    label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
                    src_label_path = os.path.join(src_subdir_labels, label_file)
                    dst_label_path = os.path.join(dst_subdir_labels, prefix + label_file)
                    if os.path.exists(src_label_path):
                        shutil.copy(src_label_path, dst_label_path)
                        pbar.update(1)
                    else:
                        print(f"Label file does not exist: {src_label_path}")

    # Copy files from dataset 1
    print(f"\nStarting to copy from {args.merge_data_1}...")
    copy_files(dataset1_path, dataset1_path, combined_dataset_path, combined_dataset_path)

    # Copy files from dataset 2 with prefix to avoid name conflicts
    print(f"\nStarting to copy from {args.merge_data_2}...")
    copy_files(dataset2_path, dataset2_path, combined_dataset_path, combined_dataset_path, prefix="dataset2_")

    # Update the data.yaml file
    class_names = ['capsicum', 'garlic', 'lemon', 'lime', 'pear', 'potato', 'pumpkin', 'tomato']

    data_yaml_path = os.path.join(combined_dataset_path, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n")
        f.write(f"nc: {len(class_names)}\n")  # Update the number of classes based on your dataset
        f.write(f"names: {class_names}\n")  # Update class names based on your dataset

def main():
    args = parse_args()
    # Determine the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths relative to the script's location
    dataset1_path = os.path.abspath(os.path.join(script_dir, '..', 'datasets', f'dataset_{args.merge_data_1}'))
    dataset2_path = os.path.abspath(os.path.join(script_dir, '..', 'datasets', f'dataset_{args.merge_data_2}'))
    combined_dataset_path = os.path.abspath(os.path.join(script_dir, '..', 'datasets', f'dataset_{args.output_data_id}'))
    
    print(f"Dataset1 Path: {dataset1_path}")
    print(f"Dataset2 Path: {dataset2_path}")
    print(f"Combined Dataset Path: {combined_dataset_path}")

    combine_datasets(args, dataset1_path, dataset2_path, combined_dataset_path)

if __name__ == '__main__':
    main()