
import os
import tensorflow as tf
from tensorflow.keras import layers

# Load the trained model
model = tf.keras.models.load_model('accident_detection_model.h5')

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Make predictions on new images
test_data_dir = './dataset/preprocessed/test'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

predictions = model.predict(test_generator)
predicted_labels = [1 if pred > 0.5 else 0 for pred in predictions]
