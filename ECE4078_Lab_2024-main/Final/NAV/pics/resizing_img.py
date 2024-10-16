
from PIL import Image
import os


def resize_image_50_percent(image_path, scaling_factor):
    # Open the image
    img = Image.open(image_path)
    
    # Resize the image by 50%
    new_size = (int(img.width * scaling_factor), int(img.height * scaling_factor))  # 50% larger
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Get the directory and the file name
    img_dir = os.path.dirname(image_path)
    img_name = os.path.basename(image_path)
    
    # Save the resized image with the same name, overwriting the original
    resized_img_path = os.path.join(img_dir, img_name)
    resized_img.save(resized_img_path)
    
    print(f'Image resized and saved as {resized_img_path}')

'''

resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/capsicum.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/garlic.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/lemon.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/lime.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/pear.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/potato.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/pumpkin.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/tomato.png', 1.5)
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit_fruit/unknown.png', 1.5) 


resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_1.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_2.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_3.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_4.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_5.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_6.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_7.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_8.png', 1.5) 
resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/lm_9.png', 1.5) 

resize_image_50_percent('/Users/ahila/Desktop/ece4078/G02/Week07-08/pics/8bit/pibot_top.png', 1.5) '''


resize_image_50_percent('pics/8bit/pibot_top2.png', 1.5) 
