# Clasificador de lesiones de piel con Deep Learning

Modelo de clasificación de imágenes que identifica **4 tipos de lesiones cutáneas** usando *transfer learning* sobre **MobileNetV2**, preentrenado en ImageNet.

> ⚠️ **Proyecto académico.** No es una herramienta de diagnóstico médico ni reemplaza la evaluación de un dermatólogo.

## Clases

| Clase | Descripción |
|---|---|
| Melanoma | Cáncer de piel originado en los melanocitos |
| Carcinoma de células basales | Tipo más común de cáncer de piel |
| Sarcoma de Kaposi | Tumor de los vasos sanguíneos y linfáticos |
| Lunares benignos | Nevos sin malignidad |

## Resultados

El modelo alcanzó una **precisión en validación de aproximadamente 75–80 %** con 15 épocas de entrenamiento, usando solo las capas superiores entrenables (capas base congeladas).

## Tecnologías

- **Python**
- **TensorFlow / Keras**: MobileNetV2, data augmentation
- **scikit-learn**: matriz de confusión y reporte de clasificación
- **Pillow**: preprocesamiento de imágenes
- **Matplotlib / Seaborn**: visualización de resultados

## Cómo funciona

```
verificar_imagenes.py  →  resize_images.py  →  train_model.py  →  test_model.py
  (limpia corruptas)       (224 × 224 px)       (entrena modelo)    (evalúa métricas)
```

1. **Verificación:** elimina imágenes dañadas que romperían el entrenamiento.
2. **Preprocesamiento:** convierte todas las imágenes a RGB y 224 × 224 px, el tamaño de entrada de MobileNetV2.
3. **Entrenamiento:**
   - Split 80 % entrenamiento / 20 % validación.
   - Data augmentation (rotación, zoom, volteo horizontal) **solo en entrenamiento**.
   - MobileNetV2 congelado + capa densa de 128 neuronas con Dropout (0.3) para reducir sobreajuste.
   - Optimizador Adam (learning rate 0.0001).
4. **Evaluación:** precisión, pérdida, matriz de confusión, reporte por clase (precision, recall, F1) y ejemplos visuales de aciertos y errores.

## Estructura del proyecto

```
├── config.py               # Rutas y parámetros compartidos
├── verificar_imagenes.py   # Paso 1: elimina imágenes corruptas
├── resize_images.py        # Paso 2: redimensiona a 224×224
├── train_model.py          # Paso 3: entrena el modelo
├── test_model.py           # Paso 4: evalúa el modelo
├── requirements.txt
└── data/                   # No incluido en el repo (ver Dataset)
    ├── original/           # Imágenes originales, una carpeta por clase
    └── data_resized/       # Generada por resize_images.py
```

## Dataset

El repositorio incluye las imágenes ya preprocesadas (224 × 224 px) en `data/data_resized/`, listas para entrenar.

Si quieres partir de imágenes originales, colócalas en `data/original/` (una carpeta por clase) y ejecuta los pasos 1 y 2.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt

python verificar_imagenes.py
python resize_images.py
python train_model.py
python test_model.py
```

Las gráficas (curvas de entrenamiento, matriz de confusión y ejemplos) se guardan en `results/`.

> El entrenamiento puede tardar bastante sin GPU.

## Posibles mejoras

- **Fine-tuning:** descongelar las últimas capas de MobileNetV2 para ajustar el modelo al dominio dermatológico.
- **Balanceo de clases:** aplicar `class_weight` si alguna clase tiene menos imágenes.
- **Early stopping:** detener el entrenamiento cuando la validación deje de mejorar.
- **Interfaz web:** crear una demo con Streamlit o Gradio para subir una imagen y ver la predicción.

## Autora

**Fabiana Peña Rodríguez**, estudiante de Ingeniería de Sistemas.
GitHub: [@Fabianap001](https://github.com/Fabianap001)