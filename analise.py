import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("modelo/modelo.keras")
class_names = ['Capacitor', 'IC', 'Resistor', 'Transistor']

def analise(caminho):

    #caminho = "static/uploads/imagem.jpg"

    img = image.load_img(
        caminho,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    resultado = {
        'class_name': class_names[predicted_class],
        'confidence': round(confidence, 2),
        'predicted_class_index': int(predicted_class),
        'probabilities': []
    }

    for i, classe in enumerate(class_names):
        probabilidade = float(prediction[0][i] * 100)
        resultado['probabilities'].append({
            'class': classe,
            'probability': round(probabilidade, 2),
            'index': i
        })
    
    return resultado