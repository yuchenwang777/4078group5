import os
import ultralytics
from IPython import display
import torch
from arg_parser import parse_args

def main():

    args = parse_args()

    # Determine the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define relative paths
    dataset_id = args.dataset_id
    # Specify a folder name for the run
    run_name = args.model_id
    batch_size = args.batch_size
    epochs = args.epochs


    train_data_path = os.path.join(script_dir, 'datasets', f'dataset_{dataset_id}', 'data.yaml')

    # Initialize the YOLO model
    model = ultralytics.YOLO("yolov8s.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", str(device))

    # Train the model
    results = model.train(
        data=train_data_path, 
        batch=batch_size, 
        epochs=epochs, 
        imgsz=320, 
        plots=True, 
        device=device,
        name=run_name,  # Set the custom name for the run
        workers=2
    )

    # Print the folder where the final weights are stored
    print(f"Saved in: {results.save_dir}")

if __name__ == '__main__':
    main()