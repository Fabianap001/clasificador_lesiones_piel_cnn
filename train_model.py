"""Paso 3: entrena un clasificador con transfer learning sobre MobileNetV2."""

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from config import (RESIZED_DIR, MODEL_DIR, MODEL_PATH, RESULTS_DIR, IMG_SIZE,
                    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT, SEED)


# Generadores de datos
# Entrenamiento: con data augmentation para que el modelo generalice mejor.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT,
)

# Validación: solo normalización. Así las métricas reflejan imágenes reales, sin alterar.
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)

train_generator = train_datagen.flow_from_directory(
    RESIZED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    seed=SEED,
)

val_generator = val_datagen.flow_from_directory(
    RESIZED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    seed=SEED,
)

# Las clases se toman de las carpetas (Keras las ordena alfabéticamente),
# así los nombres siempre coinciden con los índices que usa el modelo.
class_names = list(train_generator.class_indices.keys())
print(f"Clases detectadas: {class_names}")


# Modelo base (MobileNetV2 preentrenado en ImageNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # Se congelan las capas base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# Entrenamiento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1,
)


# Guardar el modelo entrenado
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Modelo guardado en '{MODEL_PATH}'")


# Curvas de entrenamiento
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "curvas_entrenamiento.png"), dpi=150)
plt.show()