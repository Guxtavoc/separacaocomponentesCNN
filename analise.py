import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("modelo/modelo.keras")

class_names = ['Capacitor', 'IC', 'Resistor', 'Transistor']
img_path = "teste/resistor.jpg"

img = image.load_img(
    img_path,
    target_size=(128, 128)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Classe prevista: {class_names[predicted_class]}")
print(f"Confiança: {confidence:.2f}%")
for i, classe in enumerate(class_names):
    print(f"{classe}: {prediction[0][i] * 100:.2f}%")