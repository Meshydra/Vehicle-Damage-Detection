#Pre processes the image and defines the model
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers


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

        # Save the preprocessed image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, normalized_image * 255)


# Path to the directory containing the vehicle images
# test
images_dir = './dataset/test'
output_dir = './dataset/preprocessed/test'
target_size = (224, 224)
preprocess_images(images_dir, output_dir, target_size)

# val
images_dir = './dataset/val'
output_dir = './dataset/preprocessed/val'
target_size = (224, 224)
preprocess_images(images_dir, output_dir, target_size)

# train
images_dir = './dataset/train'
output_dir = './dataset/preprocessed/train'
target_size = (224, 224)
preprocess_images(images_dir, output_dir, target_size)

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Binary classification: accident or non-accident
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

# Print the model summary
model.summary()

# Set the paths to the preprocessed images and other training parameters
train_data_dir = './dataset/preprocessed/train'
val_data_dir = './dataset/preprocessed/val'
batch_size = 32
epochs = 10

# Prepare the training and validation datasets using the ImageDataGenerator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size
)

# Save the trained model
model.save('accident_detection_model.h5')

# Print training history
print("Training completed.")
