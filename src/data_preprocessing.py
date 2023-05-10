import cv2
import os


def preprocess_images(images_dir, output_dir, target_size):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of image filenames in the directory
    image_filenames = os.listdir(images_dir)

    # Iterate over the image filenames
    for filename in image_filenames:
        # Construct the full path to the image
        image_path = os.path.join(images_dir, filename)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, target_size)

        # Normalize the pixel values to [0, 1]
        normalized_image = resized_image / 255.0

        # Apply additional preprocessing as needed
        # For example, you can convert the image to grayscale, apply data augmentation, etc.

        # Save the preprocessed image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, normalized_image * 255)


# Path to the directory containing the vehicle images
#test
images_dir = './dataset/test'
output_dir = './dataset/preprocessed/test'

target_size = (224, 224)

preprocess_images(images_dir, output_dir, target_size)

#val
images_dir = './dataset/val'
output_dir = './dataset/preprocessed/val'

target_size = (224, 224)

preprocess_images(images_dir, output_dir, target_size)
#train
images_dir = './dataset/train'
output_dir = './dataset/preprocessed/train'

target_size = (224, 224)

preprocess_images(images_dir, output_dir, target_size)

# Completion message
print("Preprocessed")
# Close any open windows
cv2.destroyAllWindows()
