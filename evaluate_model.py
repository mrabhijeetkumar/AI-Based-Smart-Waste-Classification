import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================
# CONFIGURATION
# =====================================

DATA_DIR = Path("data")
MODEL_PATH = Path("models/best_model.keras")

IMG_SIZE = (160, 160)
BATCH_SIZE = 16

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# =====================================
# LOAD TEST DATASET
# =====================================

print("\nLoading Test Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

print("\nLoading Best Trained Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# =====================================
# MODEL EVALUATION
# =====================================

print("\nEvaluating Model on Test Dataset...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# =====================================
# GENERATE PREDICTIONS
# =====================================

print("\nGenerating Predictions...")

y_true = np.concatenate([
    labels.numpy().argmax(axis=1)
    for _, labels in test_ds
])

predictions = model.predict(test_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# =====================================
# CLASSIFICATION REPORT
# =====================================

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

# =====================================
# SAVE REPORTS
# =====================================

Path("reports").mkdir(exist_ok=True)

with open("reports/classification_report.txt", "w") as file:
    file.write("SMART WASTE CLASSIFICATION REPORT\n\n")
    file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    file.write(f"Test Loss: {test_loss:.4f}\n\n")
    file.write(report)

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(10, 8))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Smart Waste Classification - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nEvaluation Completed Successfully!")
print("Classification Report: reports/classification_report.txt")
print("Confusion Matrix: reports/confusion_matrix.png")
