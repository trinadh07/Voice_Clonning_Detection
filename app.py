
from pathlib import Path
import os
import uuid

import librosa
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Application configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "voice_deepfake_cnn.keras"
UPLOAD_DIR = BASE_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
SAMPLE_RATE = 16000
DURATION = 4
N_MFCC = 40
_model = None

# Load model once when the application starts
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}. "
                "Train the model using main.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model
    
# File validation
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# Audio preprocessing
def extract_features(audio_path):
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True)
    if audio.size == 0:
        raise ValueError("The uploaded audio contains no readable samples.")
    target_length = SAMPLE_RATE * DURATION
    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
            mode="constant"
        )
    else:
        audio = audio[:target_length]
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    features = mfcc.astype(np.float32)
    model = get_model()
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    expected_height = input_shape[1]
    expected_width = input_shape[2] if len(input_shape) >= 3 else None

    if expected_height is not None and features.shape[0] != expected_height:
        raise ValueError(
            f"Model expects {expected_height} feature rows, "
            f"but extraction produced {features.shape[0]}."
        )

    if expected_width is not None and features.shape[1] != expected_width:
        if features.shape[1] > expected_width:
            features = features[:, :expected_width]
        else:
            features = np.pad(
                features,
                ((0, 0), (0, expected_width - features.shape[1])),
                mode="constant"
            )

    features = np.expand_dims(features, axis=0)
    if len(input_shape) == 4:
        features = np.expand_dims(features, axis=-1)

    return features.astype(np.float32)

def interpret_prediction(model, features):
    output = np.asarray(model.predict(features, verbose=0))

    if output.size == 0:
        raise ValueError("The model returned an empty prediction.")
    if output.size == 1:
        ai_score = float(output.reshape(-1)[0])
    elif output.shape[-1] == 2:
        ai_score = float(output.reshape(-1, 2)[0][1])

    else:
        raise ValueError(
            "Unexpected model output shape. Check the final layer "
            "and class encoding in main.py."
        )

    ai_score = float(np.clip(ai_score, 0.0, 1.0))
    real_score = 1.0 - ai_score

    prediction = (
        "AI-generated"
        if ai_score >= 0.5
        else "Real voice"
    )

    return {
        "prediction": prediction,
        "model_score": ai_score,
        "ai_score": round(ai_score * 100, 2),
        "real_score": round(real_score * 100, 2)
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "model_exists": MODEL_PATH.exists(),
        "message": (
            "Model file found."
            if MODEL_PATH.exists()
            else "Train the model before making predictions."
        )
    })
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file was received."
        }), 400

    uploaded_file = request.files["audio"]

    if not uploaded_file or not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "Please select an audio file."
        }), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported format. Use WAV, MP3, FLAC, OGG or M4A."
        }), 400

    temporary_path = None

    try:
        safe_name = secure_filename(uploaded_file.filename)
        extension = safe_name.rsplit(".", 1)[1].lower()
        temporary_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{extension}"

        uploaded_file.save(temporary_path)

        model = get_model()
        features = extract_features(temporary_path)
        result = interpret_prediction(model, features)

        return jsonify({
            "success": True,
            **result
        })

    except FileNotFoundError as exc:
        app.logger.error("Model file error: %s", exc)

        return jsonify({
            "success": False,
            "error": "The trained model is missing. Run main.py first."
        }), 503

    except (ValueError, RuntimeError) as exc:
        app.logger.warning("Audio prediction failed: %s", exc)

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Unexpected prediction error")

        return jsonify({
            "success": False,
            "error": (
                "The audio could not be processed. Check the file format "
                "and the Flask terminal for details."
            )
        }), 500

    finally:
        if temporary_path and temporary_path.exists():
            try:
                os.remove(temporary_path)
            except OSError:
                app.logger.warning(
                    "Could not remove temporary file: %s",
                    temporary_path
                )


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "error": "File is too large. Maximum upload size is 25 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
