import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from predict import predict_image

app = FastAPI(
    title="Medical Image Classification API",
    description="API pédagogique de classification d'images radiographiques (NORMAL / PNEUMONIA). "
                 "Ne constitue pas un outil de diagnostic médical.",
    version="1.0"
)

# Autoriser les requêtes venant du frontend Angular (localhost:4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Medical AI API - projet académique",
        "endpoint": "/predict (POST)",
        "warning": "Ne constitue pas un diagnostic médical réel."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_image(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")