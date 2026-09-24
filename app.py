from flask import Flask, render_template, request
import os

import librosa
import numpy as np
import torch
import torch.nn as nn


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/tb_cnn.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# CNN MODEL
# IMPORTANT:
# This architecture must exactly match train_cnn.py
# =========================================================

class CoughCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Layer 1
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Identity(),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# =========================================================
# LOAD TRAINED CNN MODEL
# =========================================================

model = CoughCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("======================================")
print("CNN MODEL LOADED SUCCESSFULLY")
print("======================================")
print("Using device:", DEVICE)


# =========================================================
# AUDIO PREPROCESSING
# WAV → MEL SPECTROGRAM
# =========================================================

def extract_mel_spectrogram(audio_path):

    # -----------------------------------------------------
    # Load audio
    # -----------------------------------------------------

    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Check empty audio
    if audio is None or len(audio) == 0:
        raise ValueError("The uploaded audio file is empty.")

    # -----------------------------------------------------
    # Make audio exactly 4 seconds
    # -----------------------------------------------------

    target_length = 16000 * 4

    if len(audio) < target_length:

        # Pad short audio with zeros
        audio = np.pad(
            audio,
            (
                0,
                target_length - len(audio)
            )
        )

    else:

        # Cut audio longer than 4 seconds
        audio = audio[:target_length]

    # -----------------------------------------------------
    # Generate Mel Spectrogram
    # -----------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_mels=64,
        n_fft=512,
        hop_length=256
    )

    # -----------------------------------------------------
    # Convert power spectrogram to decibels
    # -----------------------------------------------------

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    mel = (
        mel - np.mean(mel)
    ) / (
        np.std(mel) + 1e-8
    )

    # -----------------------------------------------------
    # Convert NumPy array → PyTorch tensor
    # -----------------------------------------------------

    mel = torch.tensor(
        mel,
        dtype=torch.float32
    )

    # -----------------------------------------------------
    # Add dimensions
    #
    # Before:
    # [64, time]
    #
    # After:
    # [1, 1, 64, time]
    #
    # 1st dimension = batch
    # 2nd dimension = channel
    # -----------------------------------------------------

    mel = mel.unsqueeze(0).unsqueeze(0)

    return mel


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    error = None

    if request.method == "POST":

        # -------------------------------------------------
        # Check whether audio file exists
        # -------------------------------------------------

        if "audio" not in request.files:

            error = "Please select a WAV audio file."

            return render_template(
                "index.html",
                result=result,
                confidence=confidence,
                error=error
            )

        file = request.files["audio"]

        # -------------------------------------------------
        # Check filename
        # -------------------------------------------------

        if file.filename == "":

            error = "Please select a WAV audio file."

            return render_template(
                "index.html",
                result=result,
                confidence=confidence,
                error=error
            )

        # -------------------------------------------------
        # Allow only WAV files
        # -------------------------------------------------

        if not file.filename.lower().endswith(".wav"):

            error = "Only WAV audio files are supported."

            return render_template(
                "index.html",
                result=result,
                confidence=confidence,
                error=error
            )

        # -------------------------------------------------
        # Save uploaded audio
        # -------------------------------------------------

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        try:

            # =================================================
            # STEP 1: AUDIO → MEL SPECTROGRAM
            # =================================================

            mel = extract_mel_spectrogram(
                filepath
            )

            # Move tensor to CPU/GPU
            mel = mel.to(DEVICE)

            # =================================================
            # STEP 2: CNN PREDICTION
            # =================================================

            with torch.no_grad():

                outputs = model(mel)

                # Convert model output to probabilities
                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )[0]

                # Get predicted class
                prediction = torch.argmax(
                    probabilities
                ).item()

                # Get model classification confidence
                confidence = round(
                    float(
                        probabilities[prediction]
                    ) * 100,
                    2
                )

            # =================================================
            # STEP 3: RESULT
            # =================================================

            if prediction == 1:

                result = (
                    "TB-associated cough pattern detected"
                )

            else:

                result = (
                    "Healthy-associated cough pattern detected"
                )

        except Exception as e:

            error = (
                "Error processing audio: "
                + str(e)
            )

        finally:

            # -------------------------------------------------
            # Delete uploaded file after processing
            # -------------------------------------------------

            if os.path.exists(filepath):

                os.remove(filepath)

    # -----------------------------------------------------
    # Display webpage
    # -----------------------------------------------------

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error
    )


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )