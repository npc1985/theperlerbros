import cv2
import numpy as np
import os
from PIL import Image

def remove_green_screen(input_path, output_path, sensitivity=50):
    """
    Remove green screen background from an image
    sensitivity: Higher = more aggressive removal (try 30-50)
    """
    # Read the image
    image = cv2.imread(input_path)
    
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color in HSV
    # Green screen typically has these HSV ranges
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply some morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur to soften edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Convert to PIL for transparency handling
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Convert mask to PIL
    mask_pil = Image.fromarray(255 - mask)  # Invert mask
    
    # Add alpha channel (transparency)
    image_rgba = image_pil.convert("RGBA")
    
    # Apply mask as alpha channel
    image_rgba.putalpha(mask_pil)
    
    # Save as PNG to preserve transparency
    image_rgba.save(output_path, "PNG")
    print(f"Processed: {input_path} -> {output_path}")

def batch_remove_backgrounds(input_folder, output_folder):
    """
    Process all images in a folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Process each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            # Change extension to .png for transparency
            output_filename = os.path.splitext(filename)[0] + '-nobg.png'
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                remove_green_screen(input_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Usage example:
if __name__ == "__main__":
    # Set your paths
    input_folder = "images"  # Your original images folder
    output_folder = "images/no-background"  # Where to save processed images
    
    # Run the batch processing
    batch_remove_backgrounds(input_folder, output_folder)
    
    print("Batch processing complete!")
    
    # If you want to process just one image for testing:
    # remove_green_screen("images/test-image.jpg", "images/test-output.png")