import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


# Configuración inicial

data_dir = "data/data_resized"
model_path = "models/mobilenet_skin.h5"

img_size = (224, 224)
batch_size = 32
class_names = ["melanoma", "carcinoma_celulas_basales", "sarcoma_kaposi", "lunares_benignos"]

print("🔍 Cargando modelo entrenado...")
model = load_model(model_path)
print("✅ Modelo cargado correctamente.")


# Preparar los datos de validación

datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical",
    shuffle=False
)


# Evaluación del modelo

print("\n📊 Evaluando modelo en el conjunto de validación...")
loss, acc = model.evaluate(val_generator)
print(f"✅ Precisión en validación: {acc*100:.2f}%")
print(f"📉 Pérdida (Loss): {loss:.4f}\n")


# Predicciones

print("🔮 Generando predicciones...")
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_generator.classes


# Matriz de confusión

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()


# Reporte de clasificación
print("\n📄 Reporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names))


# Mostrar ejemplos visuales (con color en predicción)

print("\n🖼️ Mostrando ejemplos de predicciones...")

x, y = next(val_generator)

plt.figure(figsize=(12, 8))
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    img = x[i]
    true_label = class_names[np.argmax(y[i])]
    pred_probs = model.predict(np.expand_dims(img, axis=0))
    pred_label = class_names[np.argmax(pred_probs)]

    # ✅ Verde si acierta / ❌ Rojo si falla
    color = "green" if true_label == pred_label else "red"

    plt.imshow(img)
    plt.title(f"Real: {true_label}\nPred: {pred_label}", color=color, fontsize=10, fontweight="bold")
    plt.axis("off")

plt.tight_layout()
plt.show()
