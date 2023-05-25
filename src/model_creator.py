# This file will include the implementation of the deep learning model for damage detection, such as a Convolutional Neural Network (CNN).
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers

# Define the CNN model architecture
# Define the CNN model architecture
model = tf.keras.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten the feature maps
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    # Adjust the output units to match the number of classes (2)
    layers.Dense(2, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Set the paths for your preprocessed data
train_dir = './dataset/preprocessed/train'
val_dir = './dataset/preprocessed/val'
test_dir = './dataset/preprocessed/test'

# Set the image dimensions and batch size
image_size = (224, 224)
batch_size = 32

# Create ImageDataGenerator instances for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Prepare the training, validation, and test datasets
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_dataset = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_dataset = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Train the model
epochs = 1

model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset
)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Save the trained model
model.save('damage_detection_model.h5')
