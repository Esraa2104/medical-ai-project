import os
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_NAME = "model_v3_mobilenet_weighted.keras"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", MODEL_NAME)
IMG_SIZE = (224, 224)
CLASSES = ["NORMAL", "PNEUMONIA"]

_model = None  # chargé une seule fois (lazy loading)

def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prend une image PIL et la transforme au format attendu par le modèle."""
    img = image.convert("L")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[..., np.newaxis]
    arr = np.repeat(arr, 3, axis=-1)
    arr = preprocess_input(arr * 255.0)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(image: Image.Image) -> dict:
    """Retourne la prédiction sous forme de dictionnaire."""
    model = get_model()
    arr = preprocess_image(image)
    proba_pneumonia = float(model.predict(arr, verbose=0)[0][0])
    proba_normal = 1.0 - proba_pneumonia

    prediction = CLASSES[1] if proba_pneumonia > 0.5 else CLASSES[0]

    return {
        "prediction": prediction,
        "probability_normal": round(proba_normal, 4),
        "probability_pneumonia": round(proba_pneumonia, 4),
        "model_name": MODEL_NAME
    }

if __name__ == "__main__":
    test_image_path = os.path.join(os.path.dirname(__file__), "..", "data", "test", "PNEUMONIA")
    first_image = os.listdir(test_image_path)[0]
    img = Image.open(os.path.join(test_image_path, first_image))
    result = predict_image(img)
    print(f"Image testée: {first_image}")
    print(result)