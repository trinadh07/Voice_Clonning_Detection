import sys
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "voice_deepfake_cnn.keras"
SAMPLE_RATE = 16000
DURATION = 4
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256
MAX_LENGTH = SAMPLE_RATE * DURATION


def extract_mfcc(file_path):
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

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


def predict(file_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train the model first."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    model = load_model(MODEL_PATH)

    mfcc = extract_mfcc(file_path)
    model_input = mfcc[np.newaxis, ..., np.newaxis]

    probability_fake = float(
        model.predict(model_input, verbose=0)[0][0]
    )

    if probability_fake >= 0.5:
        label = "AI-GENERATED / FAKE"
        confidence = probability_fake * 100
    else:
        label = "REAL HUMAN VOICE"
        confidence = (1 - probability_fake) * 100

    print("\n" + "=" * 50)
    print("AI VOICE DEEPFAKE DETECTION RESULT")
    print("=" * 50)
    print(f"File: {file_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Fake probability: {probability_fake:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)

    predict(sys.argv[1])
