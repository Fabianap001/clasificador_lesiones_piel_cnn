"""Paso 2: redimensiona las imágenes originales a 224x224 (tamaño de entrada de MobileNetV2)."""

import os
from PIL import Image

from config import RAW_DIR, RESIZED_DIR, IMG_SIZE, VALID_EXTENSIONS

if not os.path.isdir(RAW_DIR):
    raise FileNotFoundError(
        f"No se encontró la carpeta '{RAW_DIR}'. "
        "Coloca allí las imágenes originales, una carpeta por clase (ver README)."
    )

os.makedirs(RESIZED_DIR, exist_ok=True)

for folder in sorted(os.listdir(RAW_DIR)):
    folder_path = os.path.join(RAW_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    output_folder = os.path.join(RESIZED_DIR, folder)
    os.makedirs(output_folder, exist_ok=True)

    print(f"🔄 Procesando carpeta: {folder}...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(VALID_EXTENSIONS):
            try:
                img_path = os.path.join(folder_path, filename)
                with Image.open(img_path) as img:
                    img_resized = img.convert("RGB").resize(IMG_SIZE)
                save_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
                img_resized.save(save_path, "JPEG", quality=95)
            except Exception as e:
                print(f"⚠️ Error con {filename}: {e}")

print("\n✅ Todas las imágenes han sido redimensionadas correctamente.")
print(f"📁 Guardadas en: {os.path.abspath(RESIZED_DIR)}")