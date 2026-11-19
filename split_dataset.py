
import random
import shutil
from pathlib import Path

# Dataset paths
SOURCE_DIR = Path("dataset-resized")
OUTPUT_DIR = Path("data")

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Reproducibility
random.seed(42)

if not SOURCE_DIR.exists():
    raise FileNotFoundError("dataset-resized folder not found!")

# Waste categories
classes = [
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
]

if not classes:
    raise ValueError("No category folders found!")

total_images = 0

for class_folder in sorted(classes):

    images = [
        file for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():

        destination = OUTPUT_DIR / split_name / class_folder.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(
                image_path,
                destination / image_path.name
            )

    total_images += total

    print(f"\n{class_folder.name}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")

print("\n-----------------------------")
print("Dataset splitting completed!")
print("Total images processed:", total_images)
print("Output folder:", OUTPUT_DIR)
