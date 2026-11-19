import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuration
DATA_DIR = Path("data")
IMG_SIZE = (160, 160)
BATCH_SIZE = 16

CLASS_NAMES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

# Load test dataset only
print("\nLoading Test Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

# Get true labels
y_true = np.concatenate([
    labels.numpy().argmax(axis=1)
    for _, labels in test_ds
])

# Evaluate both models
model_paths = {
    "Original Model": "models/best_model_backup.keras",
    "Fine-Tuned Model": "models/best_fine_tuned_model.keras"
}

results = {}

for model_name, model_path in model_paths.items():
    print(f"\nLoading {model_name}...")

    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(test_ds, verbose=1)

    results[model_name] = {
        "loss": loss,
        "accuracy": accuracy
    }

    print(f"\n{model_name} Results:")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    if model_name == "Fine-Tuned Model":
        predictions = model.predict(test_ds, verbose=1)
        y_pred = np.argmax(predictions, axis=1)

        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )

        print("\nFine-Tuned Classification Report:")
        print(report)

        Path("reports").mkdir(exist_ok=True)

        with open("reports/fine_tuned_classification_report.txt", "w") as file:
            file.write(report)

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

        plt.title("Fine-Tuned Model - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            "reports/fine_tuned_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

# Final comparison
print("\n========== MODEL COMPARISON ==========")

for name, result in results.items():
    print(
        f"{name}: "
        f"Accuracy = {result['accuracy'] * 100:.2f}%, "
        f"Loss = {result['loss']:.4f}"
    )

difference = (
    results["Fine-Tuned Model"]["accuracy"]
    - results["Original Model"]["accuracy"]
) * 100

print(f"\nAccuracy Difference: {difference:+.2f} percentage points")

if difference > 0:
    print("Fine-tuned model performed better on test data.")
elif difference < 0:
    print("Original model performed better on test data.")
else:
    print("Both models achieved the same test accuracy.")

print("\nComparison completed successfully!")
