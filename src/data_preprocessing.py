import cv2
import os


def preprocess_images(images_dir, output_dir, target_size):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of subdirectories in the directory
    subdirectories = [f for f in os.listdir(
        images_dir) if os.path.isdir(os.path.join(images_dir, f))]

    # Iterate over the subdirectories
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(images_dir, subdirectory)

        # Create the corresponding output subdirectory if it doesn't exist
        output_subdirectory = os.path.join(output_dir, subdirectory)
        if not os.path.exists(output_subdirectory):
            os.makedirs(output_subdirectory)

        # Get the list of image filenames in the subdirectory
        image_filenames = os.listdir(subdirectory_path)

        # Iterate over the image filenames
        for filename in image_filenames:
            # Construct the full path to the image
            image_path = os.path.join(subdirectory_path, filename)

            # Skip if the current item is a file
            if not os.path.isfile(image_path):
                continue

            # Load the image using OpenCV
            image = cv2.imread(image_path)

            # Check if the image was loaded successfully
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            # Resize the image
            resized_image = cv2.resize(image, target_size)

            # Normalize the pixel values to [0, 1]
            normalized_image = resized_image / 255.0

            # Apply additional preprocessing as needed
            # For example, you can convert the image to grayscale, apply data augmentation, etc.

            # Save the preprocessed image to the output subdirectory
            output_path = os.path.join(output_subdirectory, filename)
            cv2.imwrite(output_path, normalized_image * 255)


# Path to the directory containing the vehicle images
# Test images
images_dir = './dataset/test'
output_dir = './dataset/preprocessed/test'

target_size = (224, 224)

preprocess_images(images_dir, output_dir, target_size)

# Val images
images_dir = './dataset/val'
output_dir = './dataset/preprocessed/val'

target_size = (224, 224)

preprocess_images(images_dir, output_dir, target_size)

# Train images
images_dir = './dataset/train'
output_dir = './dataset/preprocessed/train'

target_size = (224, 224)

preprocess_images(images_dir, output_dir, target_size)

# Completion message
print("Preprocessed")
# Close any open windows
cv2.destroyAllWindows()
