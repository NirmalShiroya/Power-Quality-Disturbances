
"""

============================================================
CUSTOM CNN FOR POWER QUALITY DISTURBANCE DETECTION
GASF + CUSTOM CNN

AUTOMATIC 6-EXPERIMENT VERSION

EVENTS:
  1. Voltage Sag
     Event Type = 1
     Feature = Upstream Voltage Min(kV)

  2. Waveform Shape Change
     Event Type = 4
     Feature = THD-V Max(%)

SPLITS:
  1. 80/10/10
  2. 70/15/15
  3. 60/20/20

TOTAL EXPERIMENTS:
  2 events x 3 splits = 6 experiments

MODEL:
  Custom CNN
  Trained from scratch
  No ImageNet pretrained weights

CNN ARCHITECTURE:
  Input 224x224x3

  Block 1:
      Conv2D 32
      Conv2D 32
      MaxPooling2D

  Block 2:
      Conv2D 64
      Conv2D 64
      MaxPooling2D

  Block 3:
      Conv2D 128
      Conv2D 128
      MaxPooling2D

  Block 4:
      Conv2D 256
      Conv2D 256
      MaxPooling2D

  Classification Head:
      GlobalAveragePooling2D
      Dense 256
      Batch Normalization
      Dropout 0.5
      Dense 2 Softmax

NO EARLY STOPPING

10:1 NORMAL : EVENT SAMPLING

CLASS WEIGHTS INCLUDED

WINDOW:
  11 points
  Center point determines label

OUTPUTS PER EXPERIMENT:
  - Best model
  - Final model
  - Training history
  - Accuracy plot
  - Loss plot
  - Confusion matrix
  - ROC curve
  - Classification report
  - Test predictions
  - Class weights
  - Class distribution
  - Model complexity
  - Training time
  - Inference time

FINAL:
  ALL_EXPERIMENTS_SUMMARY.csv
============================================================
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import time
import random
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pyts.image import GramianAngularField

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Model, Input

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.utils import Sequence


warnings.filterwarnings("ignore")


# ============================================================
# 2. RANDOM SEED
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 80)
print("RANDOM SEED")
print("=" * 80)

print("Seed:", SEED)


# ============================================================
# 3. TENSORFLOW / GPU INFORMATION
# ============================================================

print("\n")
print("=" * 80)
print("TENSORFLOW INFORMATION")
print("=" * 80)

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU available:", gpus)
    print("Number of GPUs:", len(gpus))

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

    except Exception as e:

        print(
            "GPU memory-growth warning:",
            e
        )

else:

    print("No GPU detected.")
    print("Training will use CPU.")

print("=" * 80)


# ============================================================
# 4. FILE PATHS
# ============================================================

FILE_1 = (
    r"C:/Users/nirma/Downloads/"
    r"Cleaned_Data_with_Disturbance_Type_1-6.csv"
)

FILE_2 = (
    r"C:/Users/nirma/Downloads/"
    r"Cleaned_Data_with_Disturbance_Type_7-12.csv"
)


# ============================================================
# 5. MAIN OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT = os.path.join(
    os.path.dirname(FILE_1),
    "CustomCNN_GASF_All_Experiments"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

print("\nMain output directory:")
print(OUTPUT_ROOT)


# ============================================================
# 6. GLOBAL PARAMETERS
# ============================================================

WINDOW_SIZE = 11

HALF_WINDOW = 5

NORMAL_TO_EVENT_RATIO = 10

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 1e-4

GASF_METHOD = "summation"

IMAGE_SIZE = 224

INPUT_CHANNELS = 3


# ============================================================
# 7. EXPERIMENT CONFIGURATIONS
# ============================================================

EVENT_CONFIGURATIONS = {

    "Voltage_Sag": {

        "event_type": 1,

        "event_name": "Voltage Sag",

        "feature": "Upstream Voltage Min(kV)"

    },

    "Waveform_Shape_Change": {

        "event_type": 4,

        "event_name": "Waveform Shape Change",

        "feature": "THD-V Max(%)"

    }

}


SPLIT_CONFIGURATIONS = {

    "80_10_10": {

        "train": 0.80,

        "validation": 0.10,

        "test": 0.10

    },

    "70_15_15": {

        "train": 0.70,

        "validation": 0.15,

        "test": 0.15

    },

    "60_20_20": {

        "train": 0.60,

        "validation": 0.20,

        "test": 0.20

    }

}


# ============================================================
# 8. LOAD DATA
# ============================================================

print("\n")
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

print("\nLoading File 1...")

df1 = pd.read_csv(FILE_1)

print(
    "File 1 shape:",
    df1.shape
)


print("\nLoading File 2...")

if not os.path.exists(FILE_2):

    raise FileNotFoundError(
        f"File 2 not found:\n{FILE_2}"
    )

df2 = pd.read_csv(FILE_2)

print(
    "File 2 shape:",
    df2.shape
)


# ============================================================
# 9. REQUIRED COLUMNS
# ============================================================

required_columns = [

    "Disturbance Type",

    "Upstream Voltage Min(kV)",

    "THD-V Max(%)"

]


for column in required_columns:

    if column not in df1.columns:

        raise ValueError(
            f"Column '{column}' not found in File 1."
        )

    if column not in df2.columns:

        raise ValueError(
            f"Column '{column}' not found in File 2."
        )


print("\nAll required columns are available.")


# ============================================================
# 10. COMBINE DATASETS
# ============================================================

df = pd.concat(
    [df1, df2],
    ignore_index=True
)

print("\nCombined dataset shape:")

print(df.shape)


# ============================================================
# 11. CONVERT DISTURBANCE TYPE
# ============================================================

df["Disturbance Type"] = pd.to_numeric(
    df["Disturbance Type"],
    errors="coerce"
)


# ============================================================
# 12. DISPLAY ORIGINAL DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("ORIGINAL DISTURBANCE DISTRIBUTION")
print("=" * 80)

print(
    df["Disturbance Type"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 13. CREATE GASF SEQUENCE
# ============================================================

class GASFSequence(Sequence):

    def __init__(
        self,
        patterns,
        labels,
        batch_size,
        gasf_transformer,
        sample_weights=None,
        shuffle=False
    ):

        super().__init__()

        self.patterns = np.asarray(
            patterns,
            dtype=np.float32
        )

        self.labels = np.asarray(
            labels,
            dtype=np.int32
        )

        self.batch_size = batch_size

        self.gasf_transformer = gasf_transformer

        self.sample_weights = sample_weights

        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.patterns)
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                len(self.patterns)
                / self.batch_size
            )
        )


    def __getitem__(self, index):

        start = (
            index
            * self.batch_size
        )

        end = min(
            (
                index + 1
            )
            * self.batch_size,
            len(self.patterns)
        )

        batch_indices = self.indices[
            start:end
        ]

        batch_patterns = self.patterns[
            batch_indices
        ]

        batch_labels = self.labels[
            batch_indices
        ]


        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        batch_patterns = np.nan_to_num(
            batch_patterns,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        batch_patterns = np.clip(
            batch_patterns,
            -1.0,
            1.0
        )


        # ----------------------------------------------------
        # GASF
        # ----------------------------------------------------

        gasf_images = (
            self.gasf_transformer.transform(
                batch_patterns
            )
        )


        gasf_images = np.asarray(
            gasf_images,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Add channel
        # ----------------------------------------------------

        gasf_images = np.expand_dims(
            gasf_images,
            axis=-1
        )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        gasf_images = tf.image.resize(
            gasf_images,
            [
                IMAGE_SIZE,
                IMAGE_SIZE
            ],
            method="bilinear"
        ).numpy()


        gasf_images = np.asarray(
            gasf_images,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Convert [-1, 1] to [0, 255]
        # ----------------------------------------------------

        gasf_images = (
            gasf_images + 1.0
        ) * 127.5


        gasf_images = np.clip(
            gasf_images,
            0.0,
            255.0
        )


        # ----------------------------------------------------
        # Grayscale -> RGB
        # ----------------------------------------------------

        gasf_images = np.repeat(
            gasf_images,
            3,
            axis=-1
        )


        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        gasf_images = np.nan_to_num(
            gasf_images,
            nan=0.0,
            posinf=255.0,
            neginf=0.0
        )


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if self.sample_weights is not None:

            batch_weights = (
                self.sample_weights[
                    batch_indices
                ]
            )

            return (
                gasf_images,
                batch_labels,
                batch_weights
            )


        # ----------------------------------------------------
        # Validation / Test
        # ----------------------------------------------------

        return (
            gasf_images,
            batch_labels
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# 14. CREATE BINARY DATASET
# ============================================================

def create_binary_dataset(
    dataframe,
    event_type,
    feature_column
):

    data = dataframe[
        dataframe["Disturbance Type"].isin(
            [0, event_type]
        )
    ].copy()


    data[feature_column] = pd.to_numeric(
        data[feature_column],
        errors="coerce"
    )


    data = data.dropna(
        subset=[feature_column]
    ).reset_index(
        drop=True
    )


    data["Binary_Label"] = (
        data["Disturbance Type"]
        == event_type
    ).astype(int)


    return data


# ============================================================
# 15. CREATE CENTER-LABELLED WINDOWS
# ============================================================

def create_center_windows(
    data,
    feature_column
):

    values = data[
        feature_column
    ].values.astype(
        np.float32
    )


    labels = data[
        "Binary_Label"
    ].values.astype(
        np.int32
    )


    patterns = []

    pattern_labels = []


    for i in range(
        HALF_WINDOW,
        len(values) - HALF_WINDOW
    ):

        window = values[
            i - HALF_WINDOW:
            i + HALF_WINDOW + 1
        ]


        center_label = labels[i]


        if not np.all(
            np.isfinite(window)
        ):

            continue


        patterns.append(
            window.astype(
                np.float32
            )
        )


        pattern_labels.append(
            int(center_label)
        )


    patterns = np.asarray(
        patterns,
        dtype=np.float32
    )


    pattern_labels = np.asarray(
        pattern_labels,
        dtype=np.int32
    )


    return (
        patterns,
        pattern_labels
    )


# ============================================================
# 16. NORMALIZE WINDOWS
# ============================================================

def normalize_windows(
    patterns
):

    normalized_patterns = np.zeros_like(
        patterns,
        dtype=np.float32
    )


    for i in range(
        len(patterns)
    ):

        pattern = patterns[i]


        min_value = np.min(
            pattern
        )


        max_value = np.max(
            pattern
        )


        if max_value == min_value:

            normalized_patterns[i] = 0.0

        else:

            normalized_patterns[i] = (
                2.0
                *
                (
                    (
                        pattern
                        - min_value
                    )
                    /
                    (
                        max_value
                        - min_value
                    )
                )
                - 1.0
            )


    normalized_patterns = np.nan_to_num(
        normalized_patterns,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )


    normalized_patterns = np.clip(
        normalized_patterns,
        -1.0,
        1.0
    )


    return normalized_patterns


# ============================================================
# 17. 10:1 SAMPLING
# ============================================================

def sample_10_to_1(
    patterns,
    labels,
    seed=42
):

    event_indices = np.where(
        labels == 1
    )[0]


    normal_indices = np.where(
        labels == 0
    )[0]


    event_count = len(
        event_indices
    )


    available_normal_count = len(
        normal_indices
    )


    target_normal_count = min(
        event_count
        * NORMAL_TO_EVENT_RATIO,
        available_normal_count
    )


    rng = np.random.default_rng(
        seed
    )


    selected_normal_indices = (
        rng.choice(
            normal_indices,
            size=target_normal_count,
            replace=False
        )
    )


    selected_indices = np.concatenate(
        [
            selected_normal_indices,
            event_indices
        ]
    )


    rng.shuffle(
        selected_indices
    )


    sampled_patterns = patterns[
        selected_indices
    ]


    sampled_labels = labels[
        selected_indices
    ]


    return (
        sampled_patterns,
        sampled_labels
    )


# ============================================================
# 18. SPLIT DATASET
# ============================================================

def split_dataset(
    patterns,
    labels,
    train_ratio,
    validation_ratio,
    test_ratio,
    seed=42
):

    assert abs(
        train_ratio
        + validation_ratio
        + test_ratio
        - 1.0
    ) < 1e-6


    # --------------------------------------------------------
    # TRAIN + TEMP
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = (

        train_test_split(

            patterns,

            labels,

            test_size=(
                validation_ratio
                + test_ratio
            ),

            stratify=labels,

            random_state=seed

        )

    )


    # --------------------------------------------------------
    # TEMP -> VALIDATION + TEST
    # --------------------------------------------------------

    relative_test_ratio = (

        test_ratio
        /
        (
            validation_ratio
            + test_ratio
        )

    )


    X_val, X_test, y_val, y_test = (

        train_test_split(

            X_temp,

            y_temp,

            test_size=relative_test_ratio,

            stratify=y_temp,

            random_state=seed

        )

    )


    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# 19. CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    y_train
):

    classes = np.unique(
        y_train
    )


    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )


    class_weights = dict(
        zip(
            classes,
            weights
        )
    )


    return {
        int(k): float(v)
        for k, v
        in class_weights.items()
    }


# ============================================================
# 20. BUILD CUSTOM CNN
# ============================================================

def build_custom_cnn():

    input_layer = Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            INPUT_CHANNELS
        ),
        name="GASF_Input"
    )


    # ========================================================
    # CNN BLOCK 1
    # ========================================================

    x = Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_32_1"
    )(input_layer)


    x = Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_32_2"
    )(x)


    x = MaxPooling2D(
        pool_size=(2, 2),
        name="MaxPooling_1"
    )(x)


    # ========================================================
    # CNN BLOCK 2
    # ========================================================

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_64_1"
    )(x)


    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_64_2"
    )(x)


    x = MaxPooling2D(
        pool_size=(2, 2),
        name="MaxPooling_2"
    )(x)


    # ========================================================
    # CNN BLOCK 3
    # ========================================================

    x = Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_128_1"
    )(x)


    x = Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_128_2"
    )(x)


    x = MaxPooling2D(
        pool_size=(2, 2),
        name="MaxPooling_3"
    )(x)


    # ========================================================
    # CNN BLOCK 4
    # ========================================================

    x = Conv2D(
        256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_256_1"
    )(x)


    x = Conv2D(
        256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2D_256_2"
    )(x)


    x = MaxPooling2D(
        pool_size=(2, 2),
        name="MaxPooling_4"
    )(x)


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    x = GlobalAveragePooling2D(
        name="Global_Average_Pooling"
    )(x)


    # ========================================================
    # FULLY CONNECTED CLASSIFICATION HEAD
    # ========================================================

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(0.01),
        name="Dense_256"
    )(x)


    x = BatchNormalization(
        name="Batch_Normalization"
    )(x)


    x = Dropout(
        0.5,
        name="Dropout"
    )(x)


    # ========================================================
    # OUTPUT
    # ========================================================

    output_layer = Dense(
        2,
        activation="softmax",
        name="Output"
    )(x)


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = Model(
        inputs=input_layer,
        outputs=output_layer,
        name="GASF_Custom_CNN"
    )


    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model


# ============================================================
# 21. SAVE TRAINING PLOTS
# ============================================================

def save_training_plots(
    history,
    output_directory,
    event_name,
    split_name
):

    epochs = range(
        1,
        len(
            history.history["loss"]
        ) + 1
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy"
    )


    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        f"{event_name} - "
        f"{split_name} - Accuracy"
    )


    plt.legend()


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    accuracy_path = os.path.join(
        output_directory,
        "Training_Validation_Accuracy.png"
    )


    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        epochs,
        history.history["loss"],
        label="Training Loss"
    )


    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        f"{event_name} - "
        f"{split_name} - Loss"
    )


    plt.legend()


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    loss_path = os.path.join(
        output_directory,
        "Training_Validation_Loss.png"
    )


    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    return (
        accuracy_path,
        loss_path
    )


# ============================================================
# 22. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    output_directory,
    target_names
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        "Confusion Matrix"
    )


    plt.colorbar()


    plt.xticks(
        [0, 1],
        target_names
    )


    plt.yticks(
        [0, 1],
        target_names
    )


    plt.xlabel(
        "Predicted Label"
    )


    plt.ylabel(
        "True Label"
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )


    plt.tight_layout()


    path = os.path.join(
        output_directory,
        "Confusion_Matrix.png"
    )


    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    return cm


# ============================================================
# 23. SAVE ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    probabilities,
    output_directory,
    event_name
):

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan


    event_probabilities = (
        probabilities[:, 1]
    )


    fpr, tpr, _ = roc_curve(
        y_true,
        event_probabilities
    )


    roc_auc = auc(
        fpr,
        tpr
    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(
        fpr,
        tpr,
        label=(
            f"Custom CNN GASF "
            f"(AUC = {roc_auc:.4f})"
        )
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        f"ROC Curve - {event_name}"
    )


    plt.legend(
        loc="lower right"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    path = os.path.join(
        output_directory,
        "ROC_Curve.png"
    )


    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    return roc_auc


# ============================================================
# 24. CALCULATE MODEL COMPLEXITY
# ============================================================

def calculate_model_complexity(
    model,
    model_path
):

    total_parameters = int(
        model.count_params()
    )


    trainable_parameters = int(
        np.sum(
            [
                np.prod(
                    variable.shape
                )
                for variable
                in model.trainable_weights
            ]
        )
    )


    non_trainable_parameters = int(
        np.sum(
            [
                np.prod(
                    variable.shape
                )
                for variable
                in model.non_trainable_weights
            ]
        )
    )


    if (
        model_path is not None
        and
        os.path.exists(model_path)
    ):

        model_size_bytes = os.path.getsize(
            model_path
        )


        model_size_mb = (
            model_size_bytes
            / (1024 ** 2)
        )

    else:

        model_size_mb = np.nan


    # Approximate float32 size

    approximate_size_mb = (
        total_parameters
        * 4
        / (1024 ** 2)
    )


    return {

        "Total_Parameters":
            total_parameters,

        "Trainable_Parameters":
            trainable_parameters,

        "Non_Trainable_Parameters":
            non_trainable_parameters,

        "Saved_Model_Size_MB":
            model_size_mb,

        "Approximate_Float32_Model_Size_MB":
            approximate_size_mb

    }


# ============================================================
# 25. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    event_key,
    event_config,
    split_key,
    split_config,
    dataframe
):

    event_type = event_config[
        "event_type"
    ]


    event_name = event_config[
        "event_name"
    ]


    feature_column = event_config[
        "feature"
    ]


    train_ratio = split_config[
        "train"
    ]


    validation_ratio = split_config[
        "validation"
    ]


    test_ratio = split_config[
        "test"
    ]


    experiment_name = (
        f"{event_key}_"
        f"{split_key}"
    )


    output_directory = os.path.join(
        OUTPUT_ROOT,
        experiment_name
    )


    os.makedirs(
        output_directory,
        exist_ok=True
    )


    print("\n\n")
    print("#" * 100)


    print(
        f"STARTING EXPERIMENT: "
        f"{experiment_name}"
    )


    print("#" * 100)


    print(
        f"\nEvent: {event_name}"
    )


    print(
        f"Event type: {event_type}"
    )


    print(
        f"Feature: {feature_column}"
    )


    print(
        f"Split: "
        f"{train_ratio:.0%}/"
        f"{validation_ratio:.0%}/"
        f"{test_ratio:.0%}"
    )


    try:

        # ====================================================
        # 1. CLEAR KERAS SESSION
        # ====================================================

        tf.keras.backend.clear_session()


        # ====================================================
        # 2. CREATE BINARY DATASET
        # ====================================================

        print("\n")
        print("-" * 80)
        print("CREATING BINARY DATASET")
        print("-" * 80)


        binary_data = create_binary_dataset(
            dataframe,
            event_type,
            feature_column
        )


        print(
            "\nOriginal binary distribution:"
        )


        print(
            binary_data[
                "Binary_Label"
            ]
            .value_counts()
            .sort_index()
        )


        # ====================================================
        # 3. CREATE CENTER-LABELLED WINDOWS
        # ====================================================

        print("\n")
        print("-" * 80)
        print("CREATING 11-POINT CENTER-LABELLED WINDOWS")
        print("-" * 80)


        window_start = time.time()


        patterns, pattern_labels = (
            create_center_windows(
                binary_data,
                feature_column
            )
        )


        window_time = (
            time.time()
            - window_start
        )


        print(
            "\nGenerated patterns:",
            f"{len(patterns):,}"
        )


        print(
            "Window extraction time:",
            f"{window_time:.2f} seconds"
        )


        print(
            "\nPattern distribution:"
        )


        print(
            "Normal:",
            np.sum(
                pattern_labels == 0
            )
        )


        print(
            event_name + ":",
            np.sum(
                pattern_labels == 1
            )
        )


        # ====================================================
        # 4. NORMALIZE
        # ====================================================

        print("\n")
        print("-" * 80)
        print("NORMALIZING WINDOWS TO [-1, 1]")
        print("-" * 80)


        patterns = normalize_windows(
            patterns
        )


        print(
            "Minimum:",
            np.min(patterns)
        )


        print(
            "Maximum:",
            np.max(patterns)
        )


        # ====================================================
        # 5. 10:1 SAMPLING
        # ====================================================

        print("\n")
        print("-" * 80)
        print("10:1 NORMAL : EVENT SAMPLING")
        print("-" * 80)


        available_events = np.sum(
            pattern_labels == 1
        )


        available_normals = np.sum(
            pattern_labels == 0
        )


        print(
            "\nEvent samples available:",
            available_events
        )


        print(
            "Normal samples available:",
            available_normals
        )


        patterns, pattern_labels = (
            sample_10_to_1(
                patterns,
                pattern_labels,
                SEED
            )
        )


        print(
            "\nSampled dataset:"
        )


        print(
            "Total:",
            len(patterns)
        )


        print(
            "Normal:",
            np.sum(
                pattern_labels == 0
            )
        )


        print(
            event_name + ":",
            np.sum(
                pattern_labels == 1
            )
        )


        # ====================================================
        # 6. SPLIT DATA
        # ====================================================

        print("\n")
        print("-" * 80)


        print(
            f"SPLITTING DATA "
            f"{train_ratio:.0%}/"
            f"{validation_ratio:.0%}/"
            f"{test_ratio:.0%}"
        )


        print("-" * 80)


        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = split_dataset(

            patterns,

            pattern_labels,

            train_ratio,

            validation_ratio,

            test_ratio,

            SEED

        )


        print(
            "\nTraining samples:",
            len(X_train)
        )


        print(
            "Validation samples:",
            len(X_val)
        )


        print(
            "Test samples:",
            len(X_test)
        )


        print(
            "\nTraining distribution:"
        )


        print(
            pd.Series(
                y_train
            )
            .value_counts()
            .sort_index()
        )


        print(
            "\nValidation distribution:"
        )


        print(
            pd.Series(
                y_val
            )
            .value_counts()
            .sort_index()
        )


        print(
            "\nTest distribution:"
        )


        print(
            pd.Series(
                y_test
            )
            .value_counts()
            .sort_index()
        )


        # ====================================================
        # 7. CLASS WEIGHTS
        # ====================================================

        class_weights = (
            calculate_class_weights(
                y_train
            )
        )


        print(
            "\nClass weights:"
        )


        print(
            class_weights
        )


        train_sample_weights = np.array(

            [

                class_weights[
                    int(label)
                ]

                for label in y_train

            ],

            dtype=np.float32

        )


        # ====================================================
        # 8. GASF TRANSFORMER
        # ====================================================

        gasf_transformer = (
            GramianAngularField(
                image_size=WINDOW_SIZE,
                method=GASF_METHOD
            )
        )


        # ====================================================
        # 9. CREATE SEQUENCES
        # ====================================================

        train_sequence = GASFSequence(

            patterns=X_train,

            labels=y_train,

            batch_size=BATCH_SIZE,

            gasf_transformer=gasf_transformer,

            sample_weights=train_sample_weights,

            shuffle=True

        )


        val_sequence = GASFSequence(

            patterns=X_val,

            labels=y_val,

            batch_size=BATCH_SIZE,

            gasf_transformer=gasf_transformer,

            sample_weights=None,

            shuffle=False

        )


        test_sequence = GASFSequence(

            patterns=X_test,

            labels=y_test,

            batch_size=BATCH_SIZE,

            gasf_transformer=gasf_transformer,

            sample_weights=None,

            shuffle=False

        )


        print(
            "\nTraining batches:",
            len(train_sequence)
        )


        print(
            "Validation batches:",
            len(val_sequence)
        )


        print(
            "Test batches:",
            len(test_sequence)
        )


        # ====================================================
        # 10. CHECK ONE BATCH
        # ====================================================

        sample_batch = (
            train_sequence[0]
        )


        print(
            "\nSample GASF batch:"
        )


        print(
            "Image shape:",
            sample_batch[0].shape
        )


        print(
            "Label shape:",
            sample_batch[1].shape
        )


        print(
            "Weight shape:",
            sample_batch[2].shape
        )


        # ====================================================
        # 11. BUILD CUSTOM CNN
        # ====================================================

        print("\n")
        print("-" * 80)
        print("BUILDING CUSTOM CNN")
        print("-" * 80)


        model = build_custom_cnn()


        print(
            "\nCustom CNN model created."
        )


        print(
            "Training from scratch: TRUE"
        )


        print(
            "ImageNet pretrained weights: FALSE"
        )


        # ====================================================
        # 12. MODEL SUMMARY
        # ====================================================

        model.summary()


        # ====================================================
        # 13. MODEL COMPLEXITY BEFORE TRAINING
        # ====================================================

        total_params = int(
            model.count_params()
        )


        trainable_params = int(
            np.sum(
                [
                    np.prod(
                        variable.shape
                    )

                    for variable
                    in model.trainable_variables

                ]
            )
        )


        non_trainable_params = (
            total_params
            - trainable_params
        )


        approximate_model_size_mb = (
            total_params
            * 4
            / (1024 ** 2)
        )


        print("\n")
        print("-" * 80)
        print("MODEL COMPLEXITY")
        print("-" * 80)


        print(
            "Total parameters:",
            f"{total_params:,}"
        )


        print(
            "Trainable parameters:",
            f"{trainable_params:,}"
        )


        print(
            "Non-trainable parameters:",
            f"{non_trainable_params:,}"
        )


        print(
            "Approximate model size:",
            f"{approximate_model_size_mb:.2f} MB"
        )


        # ====================================================
        # 14. CALLBACKS
        # ====================================================

        checkpoint_path = os.path.join(

            output_directory,

            "best_custom_cnn_gasf_model.keras"

        )


        reduce_lr = ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=4,

            min_lr=1e-7,

            verbose=1

        )


        checkpoint = ModelCheckpoint(

            checkpoint_path,

            monitor="val_loss",

            save_best_only=True,

            mode="min",

            verbose=1

        )


        callbacks = [

            reduce_lr,

            checkpoint

        ]


        # ====================================================
        # 15. TRAIN
        # ====================================================

        print("\n")
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)


        print(
            "\nCustom CNN is trained from scratch."
        )


        print(
            "Early stopping: DISABLED."
        )


        print(
            f"Epochs: {EPOCHS}"
        )


        training_start_time = (
            time.perf_counter()
        )


        history = model.fit(

            train_sequence,

            validation_data=val_sequence,

            epochs=EPOCHS,

            callbacks=callbacks,

            verbose=1

        )


        training_end_time = (
            time.perf_counter()
        )


        training_time_seconds = (
            training_end_time
            - training_start_time
        )


        training_time_minutes = (
            training_time_seconds
            / 60
        )


        training_time_hours = (
            training_time_seconds
            / 3600
        )


        actual_epochs = len(
            history.history["loss"]
        )


        average_training_time_per_epoch = (

            training_time_seconds
            / actual_epochs

        )


        # ====================================================
        # 16. LOAD BEST MODEL
        # ====================================================

        print("\n")
        print("-" * 80)
        print("LOADING BEST MODEL")
        print("-" * 80)


        if os.path.exists(
            checkpoint_path
        ):

            model = tf.keras.models.load_model(
                checkpoint_path
            )


            print(
                "Best model loaded successfully."
            )

        else:

            print(
                "Best model not found."
            )


        # ====================================================
        # 17. SAVE TRAINING HISTORY
        # ====================================================

        history_df = pd.DataFrame(
            history.history
        )


        history_path = os.path.join(
            output_directory,
            "training_history.csv"
        )


        history_df.to_csv(
            history_path,
            index=False
        )


        # ====================================================
        # 18. TRAINING PLOTS
        # ====================================================

        accuracy_plot_path, loss_plot_path = (

            save_training_plots(

                history,

                output_directory,

                event_name,

                split_key

            )

        )


        # ====================================================
        # 19. TRAIN PREDICTIONS
        # ====================================================

        print("\n")
        print("-" * 80)
        print("TRAINING CLASSIFICATION")
        print("-" * 80)


        train_eval_sequence = GASFSequence(

            patterns=X_train,

            labels=y_train,

            batch_size=BATCH_SIZE,

            gasf_transformer=gasf_transformer,

            sample_weights=None,

            shuffle=False

        )


        train_probabilities = model.predict(

            train_eval_sequence,

            verbose=1

        )


        train_predictions = np.argmax(

            train_probabilities,

            axis=1

        )


        train_accuracy = accuracy_score(

            y_train,

            train_predictions

        )


        train_report = classification_report(

            y_train,

            train_predictions,

            target_names=[

                "Normal",

                event_name

            ],

            digits=4,

            zero_division=0

        )


        print(
            train_report
        )


        # ====================================================
        # 20. VALIDATION PREDICTIONS
        # ====================================================

        print("\n")
        print("-" * 80)
        print("VALIDATION CLASSIFICATION")
        print("-" * 80)


        validation_probabilities = (

            model.predict(

                val_sequence,

                verbose=1

            )

        )


        validation_predictions = (

            np.argmax(

                validation_probabilities,

                axis=1

            )

        )


        validation_accuracy = (

            accuracy_score(

                y_val,

                validation_predictions

            )

        )


        validation_report = (

            classification_report(

                y_val,

                validation_predictions,

                target_names=[

                    "Normal",

                    event_name

                ],

                digits=4,

                zero_division=0

            )

        )


        print(
            validation_report
        )


        # ====================================================
        # 21. TEST INFERENCE
        # ====================================================

        print("\n")
        print("=" * 80)
        print("MEASURING TEST INFERENCE TIME")
        print("=" * 80)


        # ----------------------------------------------------
        # Warm-up prediction
        # ----------------------------------------------------

        _ = model.predict(

            test_sequence,

            verbose=0

        )


        inference_start_time = (
            time.perf_counter()
        )


        test_probabilities = model.predict(

            test_sequence,

            verbose=0

        )


        inference_end_time = (
            time.perf_counter()
        )


        inference_time_seconds = (

            inference_end_time
            - inference_start_time

        )


        num_test_samples = len(
            X_test
        )


        inference_time_per_sample = (

            inference_time_seconds
            / num_test_samples

        )


        inference_throughput = (

            num_test_samples
            / inference_time_seconds

        )


        # ====================================================
        # 22. TEST PREDICTIONS
        # ====================================================

        y_pred = np.argmax(

            test_probabilities,

            axis=1

        )


        test_accuracy = accuracy_score(

            y_test,

            y_pred

        )


        print("\n")
        print("=" * 80)
        print("TEST RESULTS")
        print("=" * 80)


        print(

            f"\nTest Accuracy: "
            f"{test_accuracy:.4f}"

        )


        print(

            f"Test Accuracy: "
            f"{test_accuracy * 100:.2f}%"

        )


        # ====================================================
        # 23. TEST CLASSIFICATION REPORT
        # ====================================================

        target_names = [

            "Normal",

            event_name

        ]


        test_report = classification_report(

            y_test,

            y_pred,

            target_names=target_names,

            digits=4,

            zero_division=0

        )


        print("\n")
        print("=" * 80)
        print("TEST CLASSIFICATION REPORT")
        print("=" * 80)


        print(
            test_report
        )


        # ====================================================
        # 24. SAVE ALL CLASSIFICATION REPORTS
        # ====================================================

        report_path = os.path.join(

            output_directory,

            "Classification_Reports.txt"

        )


        with open(

            report_path,

            "w"

        ) as f:


            f.write(
                "TRAINING CLASSIFICATION REPORT\n"
            )


            f.write(
                "=" * 70
                + "\n"
            )


            f.write(
                train_report
            )


            f.write(

                "\n\n"
                "VALIDATION CLASSIFICATION REPORT\n"

            )


            f.write(
                "=" * 70
                + "\n"
            )


            f.write(
                validation_report
            )


            f.write(

                "\n\n"
                "TESTING CLASSIFICATION REPORT\n"

            )


            f.write(
                "=" * 70
                + "\n"
            )


            f.write(
                test_report
            )


        # ====================================================
        # 25. CONFUSION MATRIX
        # ====================================================

        cm = save_confusion_matrix(

            y_test,

            y_pred,

            output_directory,

            target_names

        )


        print("\n")
        print("=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)


        print(cm)


        # ====================================================
        # 26. ROC / AUC
        # ====================================================

        roc_auc = save_roc_curve(

            y_test,

            test_probabilities,

            output_directory,

            event_name

        )


        print("\n")
        print("=" * 80)
        print("ROC / AUC")
        print("=" * 80)


        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )


        # ====================================================
        # 27. SAVE TEST PREDICTIONS
        # ====================================================

        predictions_df = pd.DataFrame(

            {

                "True_Label":
                    y_test,

                "Predicted_Label":
                    y_pred,

                "Normal_Probability":
                    test_probabilities[:, 0],

                "Event_Probability":
                    test_probabilities[:, 1]

            }

        )


        predictions_path = os.path.join(

            output_directory,

            "Test_Predictions.csv"

        )


        predictions_df.to_csv(

            predictions_path,

            index=False

        )


        # ====================================================
        # 28. SAVE CLASS WEIGHTS
        # ====================================================

        class_weights_df = pd.DataFrame(

            {

                "Class":

                    list(
                        class_weights.keys()
                    ),

                "Class_Name":

                    [

                        "Normal"

                        if int(c) == 0

                        else event_name

                        for c
                        in class_weights.keys()

                    ],

                "Weight":

                    list(
                        class_weights.values()
                    )

            }

        )


        class_weights_path = os.path.join(

            output_directory,

            "Class_Weights.csv"

        )


        class_weights_df.to_csv(

            class_weights_path,

            index=False

        )


        # ====================================================
        # 29. SAVE CLASS DISTRIBUTION
        # ====================================================

        class_distribution_df = pd.DataFrame(

            {

                "Dataset":

                    [

                        "Full Sampled Dataset",

                        "Training",

                        "Validation",

                        "Test"

                    ],

                "Normal":

                    [

                        np.sum(

                            pattern_labels == 0

                        ),

                        np.sum(

                            y_train == 0

                        ),

                        np.sum(

                            y_val == 0

                        ),

                        np.sum(

                            y_test == 0

                        )

                    ],

                event_name:

                    [

                        np.sum(

                            pattern_labels == 1

                        ),

                        np.sum(

                            y_train == 1

                        ),

                        np.sum(

                            y_val == 1

                        ),

                        np.sum(

                            y_test == 1

                        )

                    ]

            }

        )


        distribution_path = os.path.join(

            output_directory,

            "Class_Distribution.csv"

        )


        class_distribution_df.to_csv(

            distribution_path,

            index=False

        )


        # ====================================================
        # 30. SAVE MODEL COMPLEXITY
        # ====================================================

        complexity = calculate_model_complexity(

            model,

            checkpoint_path

        )


        complexity_df = pd.DataFrame(

            {

                "Experiment":
                    [experiment_name],

                "Event":
                    [event_name],

                "Event_Type":
                    [event_type],

                "Feature":
                    [feature_column],

                "Split":
                    [split_key],

                "Train_Ratio":
                    [train_ratio],

                "Validation_Ratio":
                    [validation_ratio],

                "Test_Ratio":
                    [test_ratio],

                "Architecture":
                    [
                        "Custom CNN + GASF"
                    ],

                "GASF_Method":
                    [GASF_METHOD],

                "Window_Size":
                    [WINDOW_SIZE],

                "Normal_Event_Ratio":
                    [
                        NORMAL_TO_EVENT_RATIO
                    ],

                "Training_From_Scratch":
                    [True],

                "Total_Parameters":
                    [
                        complexity[
                            "Total_Parameters"
                        ]
                    ],

                "Trainable_Parameters":
                    [
                        complexity[
                            "Trainable_Parameters"
                        ]
                    ],

                "Non_Trainable_Parameters":
                    [
                        complexity[
                            "Non_Trainable_Parameters"
                        ]
                    ],

                "Saved_Model_Size_MB":
                    [
                        complexity[
                            "Saved_Model_Size_MB"
                        ]
                    ],

                "Approximate_Model_Size_MB":
                    [
                        complexity[
                            "Approximate_Float32_Model_Size_MB"
                        ]
                    ],

                "Training_Samples":
                    [len(X_train)],

                "Validation_Samples":
                    [len(X_val)],

                "Test_Samples":
                    [len(X_test)],

                "Training_Time_Seconds":
                    [training_time_seconds],

                "Training_Time_Minutes":
                    [training_time_minutes],

                "Training_Time_Hours":
                    [training_time_hours],

                "Average_Training_Time_Per_Epoch_Seconds":
                    [
                        average_training_time_per_epoch
                    ],

                "Test_Inference_Time_Seconds":
                    [
                        inference_time_seconds
                    ],

                "Inference_Time_Per_Sample_Seconds":
                    [
                        inference_time_per_sample
                    ],

                "Inference_Throughput_Samples_Per_Second":
                    [
                        inference_throughput
                    ],

                "Training_Accuracy":
                    [train_accuracy],

                "Validation_Accuracy":
                    [validation_accuracy],

                "Test_Accuracy":
                    [test_accuracy],

                "ROC_AUC":
                    [roc_auc],

                "Epochs_Configured":
                    [EPOCHS],

                "Epochs_Completed":
                    [actual_epochs],

                "Batch_Size":
                    [BATCH_SIZE],

                "Learning_Rate":
                    [LEARNING_RATE]

            }

        )


        complexity_path = os.path.join(

            output_directory,

            "Experiment_Metrics.csv"

        )


        complexity_df.to_csv(

            complexity_path,

            index=False

        )


        # ====================================================
        # 31. SAVE FINAL MODEL
        # ====================================================

        final_model_path = os.path.join(

            output_directory,

            "Final_Custom_CNN_GASF_Model.keras"

        )


        model.save(

            final_model_path

        )


        # ====================================================
        # 32. FINAL EXPERIMENT SUMMARY
        # ====================================================

        print("\n")
        print("#" * 100)


        print(

            f"EXPERIMENT COMPLETED: "
            f"{experiment_name}"

        )


        print("#" * 100)


        print(

            f"\nEvent: {event_name}"

        )


        print(

            f"Feature: {feature_column}"

        )


        print(

            f"Split: "
            f"{train_ratio:.0%}/"
            f"{validation_ratio:.0%}/"
            f"{test_ratio:.0%}"

        )


        print(

            f"Event windows: "
            f"{np.sum(pattern_labels == 1):,}"

        )


        print(

            f"Normal windows: "
            f"{np.sum(pattern_labels == 0):,}"

        )


        print(

            f"Training samples: "
            f"{len(X_train):,}"

        )


        print(

            f"Validation samples: "
            f"{len(X_val):,}"

        )


        print(

            f"Test samples: "
            f"{len(X_test):,}"

        )


        print(

            f"\nTraining time: "
            f"{training_time_hours:.2f} hours"

        )


        print(

            f"Average training time/epoch: "
            f"{average_training_time_per_epoch:.2f} seconds"

        )


        print(

            f"Test inference time: "
            f"{inference_time_seconds:.4f} seconds"

        )


        print(

            f"Inference/sample: "
            f"{inference_time_per_sample:.6f} seconds"

        )


        print(

            f"Inference throughput: "
            f"{inference_throughput:.2f} samples/sec"

        )


        print(

            f"\nTest accuracy: "
            f"{test_accuracy * 100:.2f}%"

        )


        print(

            f"ROC-AUC: "
            f"{roc_auc:.4f}"

        )


        print(

            f"\nTotal parameters: "
            f"{total_params:,}"

        )


        print(

            f"Trainable parameters: "
            f"{trainable_params:,}"

        )


        print(

            f"\nResults saved to:\n"
            f"{output_directory}"

        )


        # ====================================================
        # RETURN MASTER RESULT
        # ====================================================

        result = {

            "Experiment":
                experiment_name,

            "Event":
                event_name,

            "Event_Type":
                event_type,

            "Feature":
                feature_column,

            "Split":
                split_key,

            "Train_Ratio":
                train_ratio,

            "Validation_Ratio":
                validation_ratio,

            "Test_Ratio":
                test_ratio,

            "Original_Event_Observations":
                int(
                    np.sum(
                        binary_data[
                            "Binary_Label"
                        ] == 1
                    )
                ),

            "Generated_Event_Windows":
                int(
                    np.sum(
                        pattern_labels == 1
                    )
                ),

            "Sampled_Normal_Windows":
                int(
                    np.sum(
                        pattern_labels == 0
                    )
                ),

            "Sampled_Event_Windows":
                int(
                    np.sum(
                        pattern_labels == 1
                    )
                ),

            "Training_Samples":
                len(X_train),

            "Validation_Samples":
                len(X_val),

            "Test_Samples":
                len(X_test),

            "Training_Normal":
                int(
                    np.sum(
                        y_train == 0
                    )
                ),

            "Training_Event":
                int(
                    np.sum(
                        y_train == 1
                    )
                ),

            "Validation_Normal":
                int(
                    np.sum(
                        y_val == 0
                    )
                ),

            "Validation_Event":
                int(
                    np.sum(
                        y_val == 1
                    )
                ),

            "Test_Normal":
                int(
                    np.sum(
                        y_test == 0
                    )
                ),

            "Test_Event":
                int(
                    np.sum(
                        y_test == 1
                    )
                ),

            "Architecture":
                "Custom CNN + GASF",

            "Training_From_Scratch":
                True,

            "Total_Parameters":
                total_params,

            "Trainable_Parameters":
                trainable_params,

            "Non_Trainable_Parameters":
                non_trainable_params,

            "Saved_Model_Size_MB":
                complexity[
                    "Saved_Model_Size_MB"
                ],

            "Approximate_Model_Size_MB":
                approximate_model_size_mb,

            "Training_Time_Seconds":
                training_time_seconds,

            "Training_Time_Minutes":
                training_time_minutes,

            "Training_Time_Hours":
                training_time_hours,

            "Average_Training_Time_Per_Epoch_Seconds":
                average_training_time_per_epoch,

            "Inference_Time_Seconds":
                inference_time_seconds,

            "Inference_Time_Per_Sample_Seconds":
                inference_time_per_sample,

            "Inference_Throughput":
                inference_throughput,

            "Training_Accuracy":
                train_accuracy,

            "Validation_Accuracy":
                validation_accuracy,

            "Test_Accuracy":
                test_accuracy,

            "ROC_AUC":
                roc_auc,

            "Epochs_Completed":
                actual_epochs,

            "Batch_Size":
                BATCH_SIZE,

            "Learning_Rate":
                LEARNING_RATE,

            "Output_Directory":
                output_directory

        }


        return result


    except Exception as e:

        print("\n")
        print("#" * 100)


        print(

            f"EXPERIMENT FAILED: "
            f"{experiment_name}"

        )


        print("#" * 100)


        print(
            "\nError:",
            e
        )


        error_text = traceback.format_exc()


        print(
            error_text
        )


        error_path = os.path.join(

            output_directory,

            "ERROR_LOG.txt"

        )


        with open(

            error_path,

            "w"

        ) as f:

            f.write(
                error_text
            )


        return None


    finally:

        tf.keras.backend.clear_session()


# ============================================================
# 33. MAIN AUTOMATIC EXPERIMENT LOOP
# ============================================================

print("\n\n")
print("#" * 100)
print("AUTOMATIC CUSTOM CNN + GASF EXPERIMENTS")
print("#" * 100)


print(

    "\nTotal event configurations:",

    len(EVENT_CONFIGURATIONS)

)


print(

    "Total split configurations:",

    len(SPLIT_CONFIGURATIONS)

)


total_experiments = (

    len(EVENT_CONFIGURATIONS)
    *
    len(SPLIT_CONFIGURATIONS)

)


print(

    "Total experiments:",

    total_experiments

)


print(

    "\nExperiments to be executed:"

)


experiment_number = 1


for event_key in EVENT_CONFIGURATIONS:

    for split_key in SPLIT_CONFIGURATIONS:

        print(

            f"{experiment_number}. "
            f"{event_key} + "
            f"{split_key}"

        )

        experiment_number += 1


# ============================================================
# 34. RUN ALL 6 EXPERIMENTS
# ============================================================

all_results = []


experiment_counter = 1


overall_start_time = (
    time.perf_counter()
)


for event_key, event_config in (

    EVENT_CONFIGURATIONS.items()

):


    for split_key, split_config in (

        SPLIT_CONFIGURATIONS.items()

    ):


        print("\n\n")
        print("*" * 100)


        print(

            f"EXPERIMENT "
            f"{experiment_counter} "
            f"OF "
            f"{total_experiments}"

        )


        print(

            f"Event: "
            f"{event_config['event_name']}"

        )


        print(

            f"Split: "
            f"{split_key}"

        )


        print("*" * 100)


        result = run_experiment(

            event_key,

            event_config,

            split_key,

            split_config,

            df

        )


        if result is not None:

            all_results.append(
                result
            )


        experiment_counter += 1


overall_end_time = (
    time.perf_counter()
)


overall_time_seconds = (

    overall_end_time
    - overall_start_time

)


overall_time_hours = (

    overall_time_seconds
    / 3600

)


# ============================================================
# 35. SAVE MASTER SUMMARY
# ============================================================

print("\n\n")
print("#" * 100)
print("ALL EXPERIMENTS FINISHED")
print("#" * 100)


print(

    f"\nSuccessful experiments: "
    f"{len(all_results)}"
    f" / "
    f"{total_experiments}"

)


print(

    f"Total execution time: "
    f"{overall_time_hours:.2f} hours"

)


if len(all_results) > 0:


    all_results_df = pd.DataFrame(
        all_results
    )


    # --------------------------------------------------------
    # Master summary
    # --------------------------------------------------------

    summary_path = os.path.join(

        OUTPUT_ROOT,

        "ALL_EXPERIMENTS_SUMMARY.csv"

    )


    all_results_df.to_csv(

        summary_path,

        index=False

    )


    print(

        "\nMaster summary saved to:"

    )


    print(
        summary_path
    )


    # --------------------------------------------------------
    # Display important results
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("MASTER RESULTS")
    print("=" * 100)


    display_columns = [

        "Experiment",

        "Event",

        "Split",

        "Training_Samples",

        "Validation_Samples",

        "Test_Samples",

        "Total_Parameters",

        "Trainable_Parameters",

        "Training_Time_Hours",

        "Inference_Time_Per_Sample_Seconds",

        "Training_Accuracy",

        "Validation_Accuracy",

        "Test_Accuracy",

        "ROC_AUC"

    ]


    available_display_columns = [

        column

        for column
        in display_columns

        if column
        in all_results_df.columns

    ]


    print(

        all_results_df[

            available_display_columns

        ].to_string(

            index=False

        )

    )


else:

    print(

        "\nNo experiments completed successfully."

    )


# ============================================================
# 36. SAVE OVERALL EXECUTION INFORMATION
# ============================================================

execution_summary = pd.DataFrame(

    {

        "Total_Experiments":
            [total_experiments],

        "Successful_Experiments":
            [len(all_results)],

        "Failed_Experiments":
            [

                total_experiments
                - len(all_results)

            ],

        "Total_Execution_Time_Seconds":
            [overall_time_seconds],

        "Total_Execution_Time_Hours":
            [overall_time_hours],

        "TensorFlow_Version":
            [tf.__version__],

        "Random_Seed":
            [SEED],

        "Model":
            ["Custom CNN"],

        "Training_From_Scratch":
            [True],

        "Window_Size":
            [WINDOW_SIZE],

        "GASF_Method":
            [GASF_METHOD],

        "Image_Size":
            [IMAGE_SIZE],

        "Batch_Size":
            [BATCH_SIZE],

        "Epochs":
            [EPOCHS],

        "Learning_Rate":
            [LEARNING_RATE],

        "Normal_Event_Ratio":
            [NORMAL_TO_EVENT_RATIO]

    }

)


execution_summary_path = os.path.join(

    OUTPUT_ROOT,

    "Overall_Execution_Summary.csv"

)


execution_summary.to_csv(

    execution_summary_path,

    index=False

)


# ============================================================
# 37. FINAL OUTPUT STRUCTURE
# ============================================================

print("\n")
print("#" * 100)
print("OUTPUT DIRECTORY STRUCTURE")
print("#" * 100)


print(

    f"""
{OUTPUT_ROOT}

├── Voltage_Sag_80_10_10/
│   ├── best_custom_cnn_gasf_model.keras
│   ├── Final_Custom_CNN_GASF_Model.keras
│   ├── training_history.csv
│   ├── Training_Validation_Accuracy.png
│   ├── Training_Validation_Loss.png
│   ├── Confusion_Matrix.png
│   ├── ROC_Curve.png
│   ├── Classification_Reports.txt
│   ├── Test_Predictions.csv
│   ├── Class_Weights.csv
│   ├── Class_Distribution.csv
│   └── Experiment_Metrics.csv

├── Voltage_Sag_70_15_15/
│   └── [same outputs]

├── Voltage_Sag_60_20_20/
│   └── [same outputs]

├── Waveform_Shape_Change_80_10_10/
│   └── [same outputs]

├── Waveform_Shape_Change_70_15_15/
│   └── [same outputs]

├── Waveform_Shape_Change_60_20_20/
│   └── [same outputs]

├── ALL_EXPERIMENTS_SUMMARY.csv

└── Overall_Execution_Summary.csv
"""

)


# ============================================================
# 38. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 100)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 100)


print(

    "\nAll requested experiments have been processed."

)


print(

    f"Successful: "
    f"{len(all_results)} / {total_experiments}"

)


print(

    f"Total execution time: "
    f"{overall_time_hours:.2f} hours"

)


print(

    "\nMain results directory:"

)


print(
    OUTPUT_ROOT
)


print(

    "\nMaster results:"

)


print(

    os.path.join(

        OUTPUT_ROOT,

        "ALL_EXPERIMENTS_SUMMARY.csv"

    )

)


print(

    "\nOverall execution summary:"

)


print(
    execution_summary_path
)


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)