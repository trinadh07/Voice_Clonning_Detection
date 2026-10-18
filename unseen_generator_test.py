"""
Generator-held-out evaluation.

This experiment is designed to test generalisation to an AI voice generator
that was NOT included in the training data.

Default:
    Held-out generator = NaturalSpeech3

The remaining fake generators plus real_samples are used for training.
The held-out generator plus real_samples are used for testing.

This is intentionally separate from main.py so the baseline and
generalisation experiment remain easy to compare.
"""

import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
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
    ReduceLROnPlateau,
)

DATASET_PATH = "dataset"
RESULTS_PATH = "results"

HELD_OUT_GENERATOR = "NaturalSpeech3"

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


def get_audio_files(folder):
    files = []
    if not os.path.exists(folder):
        return files

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(root, filename))

    return sorted(files)


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
            audio = np.pad(
                audio,
                (0, MAX_LENGTH - len(audio)),
            )
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


def collect(files, label):
    X = []
    y = []

    for i, path in enumerate(files, 1):
        if i % 100 == 0 or i == 1:
            print(f"Processing {i}/{len(files)}")

        feature = extract_mfcc(path)
        if feature is not None:
            X.append(feature)
            y.append(label)

    return X, y


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
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    if HELD_OUT_GENERATOR not in FAKE_FOLDERS:
        raise ValueError(
            f"{HELD_OUT_GENERATOR} is not in FAKE_FOLDERS."
        )

    print(f"Held-out generator: {HELD_OUT_GENERATOR}")

    # -----------------------------
    # Training data
    # -----------------------------
    train_files = []
    train_labels = []

    real_files = get_audio_files(
        os.path.join(DATASET_PATH, REAL_FOLDER)
    )

    # For a fair generator test, use all real files in both groups.
    real_X, real_y = collect(real_files, 0)
    train_X = real_X
    train_y = real_y

    for generator in FAKE_FOLDERS:
        if generator == HELD_OUT_GENERATOR:
            continue

        files = get_audio_files(
            os.path.join(DATASET_PATH, generator)
        )

        print(f"\nTraining fake generator: {generator}")
        X_part, y_part = collect(files, 1)
        train_X.extend(X_part)
        train_y.extend(y_part)

    # -----------------------------
    # Held-out test data
    # -----------------------------
    test_X = []
    test_y = []

    held_out_files = get_audio_files(
        os.path.join(DATASET_PATH, HELD_OUT_GENERATOR)
    )

    print(f"\nHeld-out fake test files: {len(held_out_files)}")
    X_part, y_part = collect(held_out_files, 1)
    test_X.extend(X_part)
    test_y.extend(y_part)

    # Use a portion of real samples for test to avoid evaluating on every
    # training real sample. A deterministic split is used.
    rng = np.random.default_rng(42)
    real_indices = np.arange(len(real_X))
    rng.shuffle(real_indices)
    real_test_count = max(1, int(0.2 * len(real_indices)))

    for idx in real_indices[:real_test_count]:
        test_X.append(real_X[idx])
        test_y.append(real_y[idx])

    train_X = np.asarray(train_X, dtype=np.float32)[..., np.newaxis]
    test_X = np.asarray(test_X, dtype=np.float32)[..., np.newaxis]
    train_y = np.asarray(train_y, dtype=np.int32)
    test_y = np.asarray(test_y, dtype=np.int32)

    # Split a validation set from training data.
    rng = np.random.default_rng(42)
    indices = np.arange(len(train_X))
    rng.shuffle(indices)

    val_count = max(1, int(0.2 * len(indices)))
    val_idx = indices[:val_count]
    fit_idx = indices[val_count:]

    X_fit = train_X[fit_idx]
    y_fit = train_y[fit_idx]
    X_val = train_X[val_idx]
    y_val = train_y[val_idx]

    model = build_model(X_fit.shape[1:])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(
        test_X,
        verbose=0,
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "held_out_generator": HELD_OUT_GENERATOR,
        "accuracy": accuracy_score(test_y, predictions),
        "precision": precision_score(
            test_y, predictions, zero_division=0
        ),
        "recall": recall_score(
            test_y, predictions, zero_division=0
        ),
        "f1": f1_score(
            test_y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(test_y, probabilities),
    }

    print("\nGenerator-held-out results")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    output = os.path.join(
        RESULTS_PATH,
        "unseen_generator_metrics.csv",
    )

    pd.DataFrame([metrics]).to_csv(
        output,
        index=False,
    )

    print(f"\nSaved: {output}")
    print("\nConfusion matrix:")
    print(confusion_matrix(test_y, predictions))


if __name__ == "__main__":
    main()
