import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process command line arguments for YOLO training and dataset management.')

    # Define command line arguments
    parser.add_argument('--model_id', type=str, required=False, default='steven',
                        help='Identifier for the model.')
    parser.add_argument('--dataset_id', type=str, required=False, default='raw',
                        help='Identifier for the dataset.')
    parser.add_argument('--batch_size', type=int, required=False, default=12, 
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, required=False, default=20, 
                        help='Number of epochs for training.')
    parser.add_argument('--dataset_merged_id', type=str, required=False, default='merged_data', 
                        help='Identifier for the merged dataset.')
    parser.add_argument('--repeats_per_fruit', type=int, required=False, default=20, 
                        help='Number of repeats per fruit for dataset generation.')
    parser.add_argument('--test_image_name', type=str, required=False, default='test_img_1',
                        help='Image name to test')
    parser.add_argument('--new_dataset_id', type=str, required=False, default='new_dataset',
                        help='Name of new generated dataset')
    
    parser.add_argument('--merge_data_1', type=str, required=False,
                        help='Name of new generated dataset')
    parser.add_argument('--merge_data_2', type=str, required=False,
                        help='Name of new generated dataset')
    parser.add_argument('--output_data_id', type=str, required=False, 
                        help='Name of new generated dataset')
    
    parser.add_argument('--num_flipped_images', type=int, required=False, 
                        help='For data augmentation - number of images to flip')

    return parser

def parse_args():
    parser = create_arg_parser()
    return parser.parse_args()