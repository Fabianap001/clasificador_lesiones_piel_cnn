"""Paso 1: revisa las imágenes originales y elimina las que estén corruptas."""

import os
from PIL import Image

from config import RAW_DIR, VALID_EXTENSIONS

print(f"🔍 Verificando imágenes dentro de: {RAW_DIR}\n")

folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]

if not folders:
    print("⚠️ No se encontraron carpetas de clases en la ruta indicada.")
else:
    total_validas = 0
    total_eliminadas = 0

    for folder in sorted(folders):
        folder_path = os.path.join(RAW_DIR, folder)
        count_validas = 0
        count_corruptas = 0

        print(f"📂 Revisando carpeta: {folder}...")

        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(VALID_EXTENSIONS):
                    file_path = os.path.join(root, filename)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()  # Verifica que no esté dañada
                        count_validas += 1
                    except Exception:
                        print(f"❌ Imagen corrupta eliminada: {file_path}")
                        os.remove(file_path)
                        count_corruptas += 1

        print(f"✅ {folder}: {count_validas} válidas, {count_corruptas} eliminadas\n")
        total_validas += count_validas
        total_eliminadas += count_corruptas

    print("🟩 Verificación completada.")
    print(f"Total de imágenes válidas: {total_validas}")
    print(f"Total de imágenes eliminadas: {total_eliminadas}")