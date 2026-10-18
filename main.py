import os
import json
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


DATASET_PATH = "dataset"
RESULTS_PATH = "results"
MODEL_PATH = "voice_deepfake_cnn.keras"

SAMPLE_RATE = 16000
DURATION = 4
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256
MAX_LENGTH = SAMPLE_RATE * DURATION

REAL_FOLDER = "real_samples"

FAKE_FOLDERS = [
    "xTTS",
    "seedtts_files",
    "VoiceBox",
    "VALLE",
    "OpenAI",
    "PromptTTS2",
    "FlashSpeech",
    "NaturalSpeech3",
]

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

os.makedirs(RESULTS_PATH, exist_ok=True)


def get_audio_files(folder):
    files = []
    if not os.path.exists(folder):
        return files

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(root, filename))

    return sorted(files)


def analyse_dataset():
    rows = []

    real_path = os.path.join(DATASET_PATH, REAL_FOLDER)
    real_files = get_audio_files(real_path)

    rows.append({
        "Folder": REAL_FOLDER,
        "Class": "Real",
        "Number of Files": len(real_files),
    })

    for folder in FAKE_FOLDERS:
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.exists(folder_path):
            rows.append({
                "Folder": folder,
                "Class": "Fake",
                "Number of Files": len(get_audio_files(folder_path)),
            })

    df = pd.DataFrame(rows)
    output = os.path.join(RESULTS_PATH, "dataset_analysis.csv")
    df.to_csv(output, index=False)

    print("\nDataset analysis")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved: {output}")

    return df



def extract_mfcc(file_path):
    try:
        audio, _ = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        if audio.size == 0:
            return None

        audio = librosa.util.normalize(audio)

        if len(audio) < MAX_LENGTH:
            audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
        else:
            audio = audio[:MAX_LENGTH]

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        return mfcc.astype(np.float32)

    except Exception as exc:
        print(f"Skipping {file_path}: {exc}")
        return None


def load_dataset():
    features = []
    labels = []
    generators = []

    # Real
    real_path = os.path.join(DATASET_PATH, REAL_FOLDER)
    real_files = get_audio_files(real_path)
    print(f"\nReal files: {len(real_files)}")

    for i, file_path in enumerate(real_files, 1):
        if i % 100 == 0 or i == 1:
            print(f"Processing real: {i}/{len(real_files)}")

        mfcc = extract_mfcc(file_path)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(0)
            generators.append("real")

    # Fake
    for generator in FAKE_FOLDERS:
        generator_path = os.path.join(DATASET_PATH, generator)

        if not os.path.exists(generator_path):
            print(f"WARNING: {generator} folder not found.")
            continue

        files = get_audio_files(generator_path)
        print(f"\n{generator} files: {len(files)}")

        for i, file_path in enumerate(files, 1):
            if i % 100 == 0 or i == 1:
                print(f"Processing {generator}: {i}/{len(files)}")

            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(1)
                generators.append(generator)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    generators = np.asarray(generators)

    if len(X) == 0:
        raise RuntimeError(
            "No usable audio was found. Check the dataset folder structure."
        )

    return X, y, generators



def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.30),

        GlobalAveragePooling2D(),

        Dense(128, activation="relu"),
        Dropout(0.40),

        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model



def save_training_history(history):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        os.path.join(RESULTS_PATH, "training_history.csv"),
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "training_accuracy.png"),
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "training_loss.png"),
        dpi=300,
    )
    plt.close()


def save_evaluation_plots(y_test, predictions, probabilities, auc):
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("AI Voice Deepfake Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "confusion_matrix.png"),
        dpi=300,
    )
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - AI Voice Deepfake Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "roc_curve.png"),
        dpi=300,
    )
    plt.close()


def train_baseline():
    print("\n" + "=" * 70)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 70)

    X, y, generators = load_dataset()

    print(f"\nFeature array: {X.shape}")
    print(f"Labels: {y.shape}")

    X = X[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    print(f"Training:   {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Testing:    {len(X_test)}")

    model = build_model(X_train.shape[1:])
    model.summary()

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_history(history)

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    metrics = pd.DataFrame([{
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }])

    metrics.to_csv(
        os.path.join(RESULTS_PATH, "metrics.csv"),
        index=False,
    )

    with open(
        os.path.join(RESULTS_PATH, "classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            classification_report(
                y_test,
                predictions,
                target_names=["Real", "Fake"],
                zero_division=0,
            )
        )

    save_evaluation_plots(
        y_test,
        predictions,
        probabilities,
        auc,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    print(f"\nBest model saved as: {MODEL_PATH}")
    print(f"Results saved in: {RESULTS_PATH}/")


if __name__ == "__main__":
    print("=" * 70)
    print("AI-GENERATED VOICE CLONING DETECTION")
    print("=" * 70)

    analyse_dataset()

    answer = input(
        "\nStart baseline CNN training? (y/n): "
    ).strip().lower()

    if answer == "y":
        train_baseline()
    else:
        print("Training cancelled.")
