from keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model from the .h5 file
model = load_model('test.h5')

# Load and preprocess the image you want to classify
img_path = 'test1.jpg'
img = Image.open(img_path).resize((224, 224))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the image

# Make predictions
predictions = model.predict(img_array)
class_index = np.argmax(predictions)
class_labels = ['Accident! Vehicle Damage Detected', 'Not an accident. No vehicle Damaged']

print('Predicted class:', class_labels[class_index])
