import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

TAG = sys.argv[1] if len(sys.argv) > 1 else "mobilenet"
assert TAG in ("mobilenet", "efficientnet"), "use: python train_v3.py mobilenet|efficientnet"

DATA_DIR = Path("data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

HEAD_EPOCHS = 10
FINE_EPOCHS = 40
HEAD_LR = 1e-3
FINE_LR = 1e-4
UNFREEZE_LAST = 60 if TAG == "mobilenet" else 120

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", class_names=CLASS_NAMES,
        shuffle=shuffle, seed=SEED if shuffle else None)


train_ds, val_ds, test_ds = load("train", True), load("val", False), load("test", False)

train_labels = np.array([CLASS_NAMES.index(Path(p).parent.name) for p in train_ds.file_paths])
cw = np.sqrt(compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_labels))
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("Backbone:", TAG)
print("Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weights.items()})

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.10),
    layers.RandomZoom((-0.15, 0.15)),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.10),
], name="augmentation")

if TAG == "mobilenet":
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
else:
    # EfficientNetV2 has its own preprocessing built in and expects raw 0-255 pixels
    base = EfficientNetV2B0(input_shape=IMG_SIZE + (3,), include_top=False,
                            weights="imagenet", include_preprocessing=True)
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,), name="image_0_255")
x = augment(inputs)
if TAG == "mobilenet":
    x = preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs, name="waste_classifier")


def cbs(path, patience, lr_patience):
    return [
        ModelCheckpoint(path, monitor="val_loss", save_best_only=True, verbose=0),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=lr_patience, min_lr=1e-6, verbose=1),
    ]


model.compile(optimizer=tf.keras.optimizers.Adam(HEAD_LR),
              loss="categorical_crossentropy", metrics=["accuracy"])
print("\n=== Phase 1: head ===")
model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
          class_weight=class_weights, callbacks=cbs(f"models/p1_{TAG}.keras", 4, 2))

base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:
    layer.trainable = False
for layer in base.layers[-UNFREEZE_LAST:]:
    layer.trainable = not isinstance(layer, layers.BatchNormalization)

model.compile(optimizer=tf.keras.optimizers.Adam(FINE_LR),
              loss="categorical_crossentropy", metrics=["accuracy"])
print("\n=== Phase 2: fine-tune ===")
best_path = f"models/best_model_v3_{TAG}.keras"
model.fit(train_ds, validation_data=val_ds, epochs=FINE_EPOCHS,
          class_weight=class_weights, callbacks=cbs(best_path, 8, 3))

best = tf.keras.models.load_model(best_path)
test_loss, test_acc = best.evaluate(test_ds, verbose=0)
print(f"\nTEST accuracy: {test_acc * 100:.2f}%   loss: {test_loss:.4f}")

y_true = np.concatenate([y.numpy().argmax(1) for _, y in test_ds])
y_pred = best.predict(test_ds, verbose=0).argmax(1)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
cm = confusion_matrix(y_true, y_pred)
print(report)
print("Confusion matrix:\n", cm)

Path(f"reports/test_report_v3_{TAG}.txt").write_text(
    f"Backbone: {TAG}\nTest accuracy: {test_acc * 100:.2f}%\n\n{report}\n\nConfusion matrix:\n{cm}\n")
Path("models/class_names.json").write_text(json.dumps(CLASS_NAMES))
print(f"\nSaved: {best_path} and reports/test_report_v3_{TAG}.txt")
