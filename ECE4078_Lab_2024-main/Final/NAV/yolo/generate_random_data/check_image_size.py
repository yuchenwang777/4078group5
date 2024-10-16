import cv2
import os

def check_image_size(image_path):
    """
    Function to check the size (width and height in pixels) of an image.
    Input:
        image_path: str, path to the image file
    Output:
        width: int, width of the image in pixels
        height: int, height of the image in pixels
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    return width, height

if __name__ == "__main__":
    # Example usage
    image_path = "backgrounds/img_6.png"  # Replace with the path to your image file
    width, height = check_image_size(image_path)
    
    if width is not None and height is not None:
        print(f"Image size: {width}x{height} pixels")