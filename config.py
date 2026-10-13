"""Configuración compartida por todos los scripts del proyecto."""

import os

# Rutas
RAW_DIR = os.path.join("data", "original")          # imágenes originales, una carpeta por clase
RESIZED_DIR = os.path.join("data", "data_resized")  # imágenes redimensionadas (salida de resize_images.py)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_skin.h5")
RESULTS_DIR = "results"

# Parámetros de imagen y entrenamiento
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
SEED = 42

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")