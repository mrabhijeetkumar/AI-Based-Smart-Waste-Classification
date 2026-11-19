import tensorflow as tf
import numpy as np

from pathlib import Path
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight

# =====================================
# CONFIGURATION
# =====================================

DATA_DIR = Path("data")
MODEL_PATH = "models/best_model_backup.keras"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 6

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# =====================================
# LOAD TRAINING DATA
# =====================================

print("\nLoading Training Dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=True,
    seed=42
)

print("\nLoading Validation Dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =====================================
# LOAD EXISTING BEST MODEL
# =====================================

print("\nLoading Original Best Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Original Model Loaded Successfully!")

# =====================================
# UNFREEZE LAST 30 LAYERS
# =====================================

base_model = model.get_layer("mobilenetv2_1.00_160")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

print("\nMobileNetV2 Fine-Tuning Configuration:")
print("Last 30 layers selected.")
print("BatchNormalization layers kept frozen.")

# =====================================
# RECOMPILE MODEL
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Recompiled Successfully!")
print("Trainable Parameters:", model.trainable_weights.__len__())

# =====================================
# CLASS WEIGHTS
# =====================================

print("\nCalculating Class Weights...")

y_train = np.concatenate([
    labels.numpy().argmax(axis=1)
    for _, labels in train_ds
])

weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights_array)
}

print("Class Weights:", class_weights)

# =====================================
# CALLBACKS
# =====================================

Path("models").mkdir(exist_ok=True)

callbacks = [
    ModelCheckpoint(
        "models/best_fine_tuned_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )
]

# =====================================
# START FINE-TUNING
# =====================================

print("\nStarting Fine-Tuning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================
# SAVE FINAL FINE-TUNED MODEL
# =====================================

model.save("models/fine_tuned_final_model.keras")

print("\nFine-Tuning Completed Successfully!")
print("Best Fine-Tuned Model: models/best_fine_tuned_model.keras")
print("Final Fine-Tuned Model: models/fine_tuned_final_model.keras")
print("Original Backup Preserved: models/best_model_backup.keras")
