import os
import cv2

def get_image(img_name):
    # Check if the directory exists
    script_dir = os.path.abspath(os.path.dirname(__file__))
    directory = os.path.abspath(os.path.join(script_dir, '..', 'test_imgs', f'{img_name}.png'))
    
    image = cv2.imread(directory)
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image