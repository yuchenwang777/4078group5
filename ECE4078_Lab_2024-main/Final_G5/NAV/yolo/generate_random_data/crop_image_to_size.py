from PIL import Image
import os

def crop_and_save_image(image_path, output_size=(320, 240)):
    # Open the image file
    with Image.open(image_path) as img:
        # Get the dimensions of the image
        width, height = img.size

        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2

        # Calculate the cropping box
        left = max(center_x - output_size[0] // 2, 0)
        top = max(center_y - output_size[1] // 2, 0)
        right = min(center_x + output_size[0] // 2, width)
        bottom = min(center_y + output_size[1] // 2, height)

        # Ensure the box is the right size
        if right - left < output_size[0]:
            right = min(left + output_size[0], width)
        if bottom - top < output_size[1]:
            bottom = min(top + output_size[1], height)

        # Crop the image
        img_cropped = img.crop((left, top, right, bottom))

        # Resize to exact dimensions if needed (to handle cases where original dimensions are smaller)
        img_cropped = img_cropped.resize(output_size, Image.LANCZOS)

        # Save the cropped image
        img_cropped.save(image_path)

        print(f"Image cropped and saved to {image_path}")

def main():
    # Determine the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the directory containing images
    directory = os.path.join(script_dir, 'backgrounds')  # Change this to your directory path

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            crop_and_save_image(image_path)

if __name__ == '__main__':
    main()