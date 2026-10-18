# AI-Generated Voice Cloning Detection using Deep Learning

MSc Computer Science dissertation project.

This project detects real human speech versus AI-generated/deepfake speech using
MFCC acoustic features and a Convolutional Neural Network (CNN).

## Dataset expected

Place the extracted Kaggle dataset folders inside `dataset/`:

- real_samples
- xTTS
- seedtts_files
- VoiceBox
- VALLE
- OpenAI
- PromptTTS2
- FlashSpeech
- NaturalSpeech3

The code treats `real_samples` as REAL (0) and all generator folders as FAKE (1).

The dataset itself is NOT included in this ZIP.

## Project structure

AI_Voice_Deepfake_Detection/
├── dataset/
├── results/
├── templates/
│   └── index.html
├── main.py
├── predict.py
├── app.py
├── unseen_generator_test.py
├── requirements.txt
└── README.md

## Setup

Recommended Python: 3.10 or 3.11.

Windows:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Step 1: Check the dataset

Run:

    python main.py

When asked:

    Do you want to start model training? (y/n):

enter:

    n

This creates `results/dataset_analysis.csv`.

## Step 2: Train the baseline CNN

Run:

    python main.py

Enter:

    y

The program:
1. Loads audio.
2. Resamples to 16 kHz.
3. Converts to mono.
4. Normalises the waveform.
5. Pads/truncates to 4 seconds.
6. Extracts 40 MFCC coefficients.
7. Standardises MFCCs.
8. Splits into train/validation/test sets.
9. Trains a CNN.
10. Saves the best model.
11. Calculates accuracy, precision, recall, F1 and ROC-AUC.
12. Creates a confusion matrix and ROC curve.

## Step 3: Test one audio file

Put an audio file in the project folder and run:

    python predict.py test.wav

## Step 4: Run the web application

After training:

    python app.py

Open the local address shown by Flask in your browser.

## Step 5: Unseen-generator experiment

The baseline random split is useful, but it can give optimistic results when
samples from the same generators appear in both training and testing.

Run:

    python unseen_generator_test.py

This performs a generator-held-out experiment. By default it holds out
NaturalSpeech3. You can change `HELD_OUT_GENERATOR` in that file.

Important:
- This is a research experiment, not a claim that the detector will work on every
  future voice-cloning system.
- Record all actual dataset counts and experimental results in the dissertation.
- Do not use example values from documentation as dissertation results.

## Research outputs

The scripts create files in `results/`, including:
- dataset_analysis.csv
- training_history.png
- confusion_matrix.png
- roc_curve.png
- metrics.csv
- unseen_generator_metrics.csv

## Important dissertation note

The system is intentionally close to the proposed project direction:
MFCC feature extraction + CNN binary classification. It does not claim to invent
a new deep-learning architecture.

The main experimental contribution can instead be the systematic evaluation of
performance, preprocessing, feature representation, and generalisation to an
unseen voice-generation source.
