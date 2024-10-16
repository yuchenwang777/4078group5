import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arg_parser import parse_args

def count_files_in_directory(directory, extensions):
    return len([f for f in os.listdir(directory) if f.endswith(extensions)])

def print_dataset_size(dataset_path):
    subdirs = ['train', 'val', 'test']
    extensions = ('.jpg', '.png')
    label_extension = '.txt'
    
    print(f"\nDataset statistics for '{dataset_path}':")
    
    for subdir in subdirs:
        images_dir = os.path.join(dataset_path, subdir, 'images')
        labels_dir = os.path.join(dataset_path, subdir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"{subdir.capitalize()} split:")
            print(f"  - Images directory does not exist: {images_dir}")
            print(f"  - Labels directory does not exist: {labels_dir}")
            continue
        
        num_images = count_files_in_directory(images_dir, extensions)
        num_labels = count_files_in_directory(labels_dir, (label_extension,))
        
        print(f"{subdir.capitalize()} split:")
        print(f"  - Number of images: {num_images}")
        print(f"  - Number of labels: {num_labels}")

    total_images = sum(count_files_in_directory(os.path.join(dataset_path, subdir, 'images'), extensions) for subdir in subdirs)
    total_labels = sum(count_files_in_directory(os.path.join(dataset_path, subdir, 'labels'), (label_extension,)) for subdir in subdirs)
    
    print(f"Total number of images: {total_images}")
    print(f"Total number of labels: {total_labels}")

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(script_dir, '..', 'datasets', f'dataset_{args.dataset_id}'))
    
    if not os.path.isdir(dataset_path):
        print(f"Dataset directory does not exist: {dataset_path}")
        return
    
    print_dataset_size(dataset_path)

if __name__ == '__main__':
    main()