# ================================================================
# CROPAI // COMMON HIGH-ACCURACY TRAINER
# Paddy | Cotton | Sugarcane | Corn | Wheat
# ================================================================

import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)


# ================================================================
# 1. CONFIGURATION
# ================================================================

DATASET_DIR = Path("/mnt/g/CropAI/dataset/train")
MODEL_DIR   = Path("/mnt/g/CropAI/models")
OUTPUT_DIR  = Path("/mnt/g/CropAI/outputs")

IMG_SIZE = 224
BATCH_SIZE = 16

VALIDATION_SPLIT = 0.20

HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 30

SEED = 42

SUPPORTED_CROPS = [
    "Paddy",
    "Cotton",
    "Sugarcane",
    "Corn",
    "Wheat"
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 2. SELECT CROP
# ================================================================

if len(sys.argv) >= 2:

    CROP = sys.argv[1].strip().capitalize()

else:

    print("\nAVAILABLE CROPS")
    print("-" * 40)

    for i, crop in enumerate(
        SUPPORTED_CROPS,
        start=1
    ):
        print(f"{i}. {crop}")

    choice = input(
        "\nEnter crop name: "
    ).strip().capitalize()

    CROP = choice


if CROP not in SUPPORTED_CROPS:

    raise ValueError(
        f"\nUnsupported crop: {CROP}\n"
        f"Choose from: {SUPPORTED_CROPS}"
    )


print()
print("=" * 72)
print(f"CROPAI // {CROP.upper()} HIGH-ACCURACY TRAINING")
print("=" * 72)


# ================================================================
# 3. GPU CONFIGURATION
# ================================================================

gpus = tf.config.list_physical_devices(
    "GPU"
)

print("\nGPU:", gpus)


if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy(
            "mixed_float16"
        )

        print(
            "Mixed precision: ENABLED"
        )

    except Exception as e:

        print(
            "GPU configuration warning:",
            e
        )

else:

    print(
        "WARNING: Training on CPU."
    )


# ================================================================
# 4. FIND ONLY SELECTED CROP CLASSES
# ================================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_DIR}"
    )


crop_prefix = CROP.lower() + "_"


CLASS_NAMES = sorted([

    folder.name

    for folder in DATASET_DIR.iterdir()

    if (
        folder.is_dir()
        and
        folder.name.lower().startswith(
            crop_prefix
        )
    )

])


if len(CLASS_NAMES) < 2:

    raise RuntimeError(

        f"\nNot enough {CROP} classes found.\n"
        f"Expected folders beginning with:\n"
        f"{CROP}_"

    )


NUM_CLASSES = len(
    CLASS_NAMES
)


print()
print("=" * 72)
print("CLASSES DETECTED")
print("=" * 72)


for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"{i:2d} -> {name}"
    )


print(
    "\nNumber of classes:",
    NUM_CLASSES
)


# ================================================================
# 5. COLLECT IMAGES
# ================================================================

paths = []
labels = []


print()
print("=" * 72)
print("DATASET INFORMATION")
print("=" * 72)


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    folder = (
        DATASET_DIR /
        class_name
    )

    images = [

        file

        for file in folder.rglob("*")

        if (
            file.is_file()
            and
            file.suffix.lower()
            in VALID_EXTENSIONS
        )

    ]


    print(
        f"{class_name:<40} "
        f"{len(images):>6} images"
    )


    for image in images:

        paths.append(
            str(image)
        )

        labels.append(
            class_index
        )


paths = np.asarray(paths)
labels = np.asarray(
    labels,
    dtype=np.int32
)


if len(paths) == 0:

    raise RuntimeError(
        "No images were found."
    )


print()
print(
    "TOTAL IMAGES:",
    len(paths)
)


# ================================================================
# 6. TRAIN / VALIDATION SPLIT
# ================================================================

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(

        paths,
        labels,

        test_size=VALIDATION_SPLIT,

        random_state=SEED,

        stratify=labels

    )
)


print()
print(
    "TRAIN IMAGES      :",
    len(train_paths)
)

print(
    "VALIDATION IMAGES :",
    len(val_paths)
)


# ================================================================
# 7. CLASS WEIGHTS
# ================================================================

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(
        NUM_CLASSES
    ),

    y=train_labels

)


CLASS_WEIGHTS = {

    i: float(weight)

    for i, weight
    in enumerate(weights)

}


print()
print("=" * 72)
print("CLASS WEIGHTS")
print("=" * 72)


for i, weight in CLASS_WEIGHTS.items():

    print(
        f"{CLASS_NAMES[i]:40} "
        f": {weight:.4f}"
    )


# ================================================================
# 8. IMAGE DECODING
# ================================================================

def load_image(
    path,
    label
):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_image(

        image,

        channels=3,

        expand_animations=False

    )

    image.set_shape([
        None,
        None,
        3
    ])

    image = tf.image.resize(

        image,

        [
            IMG_SIZE,
            IMG_SIZE
        ]

    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ================================================================
# 9. DATA AUGMENTATION
# ================================================================

augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomTranslation(
            0.06,
            0.06
        ),

        layers.RandomContrast(
            0.12
        )

    ],

    name="crop_augmentation"

)


# ================================================================
# 10. TF.DATA PIPELINE
# ================================================================

def create_dataset(
    image_paths,
    image_labels,
    training=False
):

    ds = tf.data.Dataset.from_tensor_slices(

        (
            image_paths,
            image_labels
        )

    )


    if training:

        ds = ds.shuffle(

            min(
                len(image_paths),
                3000
            ),

            seed=SEED,

            reshuffle_each_iteration=True

        )


    ds = ds.map(

        load_image,

        num_parallel_calls=
        tf.data.AUTOTUNE

    )


    if training:

        ds = ds.map(

            lambda image, label:
            (
                augmentation(
                    image,
                    training=True
                ),
                label
            ),

            num_parallel_calls=
            tf.data.AUTOTUNE

        )


    ds = ds.batch(
        BATCH_SIZE
    )


    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )


    return ds


train_ds = create_dataset(

    train_paths,

    train_labels,

    training=True

)


val_ds = create_dataset(

    val_paths,

    val_labels,

    training=False

)


# ================================================================
# 11. EFFICIENTNETB0 TRANSFER LEARNING MODEL
# ================================================================

print()
print("=" * 72)
print("BUILDING EFFICIENTNETB0 MODEL")
print("=" * 72)


base_model = EfficientNetB0(

    weights="imagenet",

    include_top=False,

    input_shape=(

        IMG_SIZE,
        IMG_SIZE,
        3

    )

)


# Freeze pretrained network initially

base_model.trainable = False


inputs = layers.Input(

    shape=(

        IMG_SIZE,
        IMG_SIZE,
        3

    ),

    name="crop_image"

)


x = base_model(

    inputs,

    training=False

)


x = layers.GlobalAveragePooling2D(
    name="global_average_pool"
)(x)


x = layers.BatchNormalization(
    name="feature_normalization"
)(x)


x = layers.Dropout(
    0.35
)(x)


x = layers.Dense(

    256,

    activation="relu",

    name="feature_dense"

)(x)


x = layers.BatchNormalization()(x)


x = layers.Dropout(
    0.25
)(x)


outputs = layers.Dense(

    NUM_CLASSES,

    activation="softmax",

    dtype="float32",

    name="disease_prediction"

)(x)


model = Model(

    inputs,

    outputs,

    name=f"{CROP.lower()}_disease_ai"

)


# ================================================================
# 12. MODEL PATHS
# ================================================================

crop_lower = CROP.lower()


BEST_MODEL = (

    MODEL_DIR /

    f"{crop_lower}_disease_best.keras"

)


FINAL_MODEL = (

    MODEL_DIR /

    f"{crop_lower}_disease_final.keras"

)


CROP_OUTPUT = (

    OUTPUT_DIR /

    crop_lower

)


CROP_OUTPUT.mkdir(

    parents=True,

    exist_ok=True

)


# ================================================================
# 13. CALLBACKS
# ================================================================

callbacks = [

    ModelCheckpoint(

        str(BEST_MODEL),

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1

    ),


    EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=7,

        restore_best_weights=True,

        verbose=1

    ),


    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-7,

        verbose=1

    ),


    CSVLogger(

        str(

            CROP_OUTPUT /

            "training_history.csv"

        )

    )

]


# ================================================================
# 14. PHASE 1
# TRAIN CLASSIFICATION HEAD
# ================================================================

print()
print("=" * 72)
print("PHASE 1 // TRANSFER LEARNING")
print("=" * 72)


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-3

    ),

    loss=
    tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[

        "accuracy",

        tf.keras.metrics.
        SparseTopKCategoricalAccuracy(

            k=2,

            name="top2_accuracy"

        )

    ]

)


model.summary()


model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=HEAD_EPOCHS,

    class_weight=
    CLASS_WEIGHTS,

    callbacks=
    callbacks

)


# ================================================================
# 15. PHASE 2
# FINE-TUNING
# ================================================================

print()
print("=" * 72)
print("PHASE 2 // FINE TUNING")
print("=" * 72)


base_model.trainable = True


# Keep approximately first 65% frozen

fine_tune_start = int(

    len(base_model.layers)
    * 0.65

)


for layer in base_model.layers[
    :fine_tune_start
]:

    layer.trainable = False


for layer in base_model.layers[
    fine_tune_start:
]:

    layer.trainable = True


# BatchNormalization layers remain frozen

for layer in base_model.layers:

    if isinstance(

        layer,

        tf.keras.layers.BatchNormalization

    ):

        layer.trainable = False


model.compile(

    optimizer=
    tf.keras.optimizers.Adam(

        learning_rate=2e-5

    ),

    loss=
    tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[

        "accuracy",

        tf.keras.metrics.
        SparseTopKCategoricalAccuracy(

            k=2,

            name="top2_accuracy"

        )

    ]

)


model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=
    CLASS_WEIGHTS,

    callbacks=
    callbacks

)


# ================================================================
# 16. LOAD BEST MODEL
# ================================================================

print()
print("=" * 72)
print("LOADING BEST MODEL")
print("=" * 72)


best_model = tf.keras.models.load_model(

    BEST_MODEL

)


# ================================================================
# 17. VALIDATION PREDICTIONS
# ================================================================

print()
print("=" * 72)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 72)


probabilities = best_model.predict(

    val_ds,

    verbose=1

)


predictions = np.argmax(

    probabilities,

    axis=1

)


# ================================================================
# 18. CLASSIFICATION REPORT
# ================================================================

report = classification_report(

    val_labels,

    predictions,

    labels=np.arange(
        NUM_CLASSES
    ),

    target_names=
    CLASS_NAMES,

    digits=4,

    zero_division=0

)


print()
print("=" * 72)
print(
    f"{CROP.upper()} AI // CLASSIFICATION REPORT"
)
print("=" * 72)

print(report)


# ================================================================
# 19. CONFUSION MATRIX
# ================================================================

matrix = confusion_matrix(

    val_labels,

    predictions,

    labels=np.arange(
        NUM_CLASSES
    )

)


print()
print("=" * 72)
print(
    f"{CROP.upper()} AI // CONFUSION MATRIX"
)
print("=" * 72)

print(
    "Rows    = ACTUAL"
)

print(
    "Columns = PREDICTED"
)

print()

print(matrix)


# ================================================================
# 20. PER-CLASS RECALL
# ================================================================

print()
print("=" * 72)
print("PER-CLASS RECALL")
print("=" * 72)


for i, class_name in enumerate(
    CLASS_NAMES
):

    total = matrix[i].sum()

    recall = (

        matrix[i, i] / total

        if total > 0

        else 0

    )


    print(

        f"{class_name:<40} "
        f": {recall * 100:6.2f}%"

    )


# ================================================================
# 21. FINAL MODEL EVALUATION
# ================================================================

results = best_model.evaluate(

    val_ds,

    verbose=0

)


print()
print("=" * 72)
print(
    f"{CROP.upper()} AI // TRAINING COMPLETE"
)
print("=" * 72)


for name, value in zip(

    best_model.metrics_names,

    results

):

    print(

        f"{name:<20}: "
        f"{value:.5f}"

    )


# ================================================================
# 22. SAVE FINAL MODEL
# ================================================================

best_model.save(
    FINAL_MODEL
)


# ================================================================
# 23. SAVE CLASS NAMES
# Extremely important for camera inference.
# ================================================================

class_file = (

    CROP_OUTPUT /

    "classes.json"

)


with open(

    class_file,

    "w"

) as f:

    json.dump(

        CLASS_NAMES,

        f,

        indent=4

    )


# Save report

report_file = (

    CROP_OUTPUT /

    "classification_report.txt"

)


with open(

    report_file,

    "w"

) as f:

    f.write(report)


print()
print("BEST MODEL:")
print(BEST_MODEL)

print()
print("FINAL MODEL:")
print(FINAL_MODEL)

print()
print("CLASS FILE:")
print(class_file)

print()
print("REPORT:")
print(report_file)

print()
print("=" * 72)
print("CLASS ORDER USED BY MODEL")
print("=" * 72)


for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{i} -> {class_name}"
    )


print()
print(
    "Training completed successfully."
)