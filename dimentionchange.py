import cv2
import os

# Define the path to your dataset
dataset_path = '/home/armin/DDAMFN/versions/3/test/angry/'
upscaled_path = '/home/armin/DDAMFN/versions/3/test56/angry1/'

# Create the directory for saving upscaled results if it does not exist
os.makedirs(upscaled_path, exist_ok=True)

# Define the new dimensions for downscaling and the original dimensions
new_width, new_height = 56, 56
original_dimensions = (112, 112)

# Loop through each file in the dataset folder
for filename in os.listdir(dataset_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions based on your dataset
        # Construct the full path to the image file
        image_path = os.path.join(dataset_path, filename)
        
        # Load the original image
        image = cv2.imread(image_path)

        if image is None:
            continue  # Skip files that cannot be loaded as images

        # Resize to decrease the dimensions
        downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Resize back to the original dimensions
        upscaled_image = cv2.resize(downscaled_image, original_dimensions, interpolation=cv2.INTER_CUBIC)

        # Save only the upscaled image
        upscaled_filename = os.path.join(upscaled_path, filename)
        cv2.imwrite(upscaled_filename, upscaled_image)

print("Processing complete. Upscaled images are saved.")

