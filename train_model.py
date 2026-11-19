
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# =====================================
# PROJECT CONFIGURATION
# =====================================

DATA_DIR = Path("data")

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
NUM_CLASSES = 6
EPOCHS = 15
SEED = 42

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

print("TensorFlow Version:", tf.__version__)
print("Image Size:", IMG_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Number of Classes:", NUM_CLASSES)


# =====================================
# LOAD DATASETS
# =====================================

print("\nLoading Training Dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=True,
    seed=SEED
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

print("\nLoading Testing Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

# Dataset performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("\nAll datasets loaded successfully!")

print("Class Names:", CLASS_NAMES)


# =====================================
# DATA AUGMENTATION
# =====================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")


# =====================================
# BUILD MOBILENETV2 MODEL
# =====================================

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Input layer
inputs = layers.Input(shape=(160, 160, 3))

# Augmentation
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = preprocess_input(x)

# Feature extraction
x = base_model(x, training=False)

# Classification layers
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

# Final model
model = models.Model(inputs, outputs)

print("\nMobileNetV2 Model Created Successfully!")
print("Total Parameters:", model.count_params())


# =====================================
# COMPILE MODEL
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Compiled Successfully!")

print("\nModel Summary:")
model.summary()


# =====================================
# CLASS WEIGHTS
# =====================================

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.concatenate([
        labels.numpy().argmax(axis=1)
        for _, labels in train_ds
    ])
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass Weights:")

for class_name, weight in zip(CLASS_NAMES, class_weights_array):
    print(f"{class_name}: {weight:.3f}")


# =====================================
# TRAINING CALLBACKS
# =====================================

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# Create model directory
Path("models").mkdir(exist_ok=True)

callbacks = [

    ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

print("\nTraining Callbacks Created Successfully!")


# =====================================
# TRAIN MOBILENETV2 MODEL
# =====================================

print("\nStarting Model Training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save final model
model.save("models/final_model.keras")

print("\nModel Training Completed Successfully!")
print("Best Model: models/best_model.keras")
print("Final Model: models/final_model.keras")

# LOAD BEST MODEL FOR EVALUATION
best_model = tf.keras.models.load_model(
    "models/best_model.keras"
)

print("\nBest Model Loaded Successfully!")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
