import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\jagad\Downloads\archive\Kaggle_RespiScan_Dataset"

SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 64

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# MEL-SPECTROGRAM EXTRACTION
# ============================================================

def extract_melspectrogram(filepath):

    audio, sr = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio is None or len(audio) == 0:
        raise ValueError("Empty audio file")

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    else:

        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=256
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )

    return mel_db.astype(np.float32)


# ============================================================
# TEST DATASET
# ============================================================

class CoughTestDataset(Dataset):

    def __init__(self):

        self.files = []
        self.labels = []

        for label, class_name in enumerate(
            ["Healthy", "TB"]
        ):

            folder = os.path.join(
                DATASET_PATH,
                "audio_wav",
                "test",
                class_name
            )

            print(
                f"\nLoading {class_name} test files..."
            )

            for filename in os.listdir(folder):

                if not filename.lower().endswith(".wav"):
                    continue

                filepath = os.path.join(
                    folder,
                    filename
                )

                # Test the file before adding it
                try:

                    audio, sr = librosa.load(
                        filepath,
                        sr=SAMPLE_RATE,
                        mono=True
                    )

                    if audio is None or len(audio) == 0:

                        print(
                            "Skipping empty audio:",
                            filename
                        )

                        continue

                except Exception as e:

                    print(
                        "Skipping invalid audio:",
                        filename
                    )

                    continue

                self.files.append(filepath)
                self.labels.append(label)

        print(
            "\nTotal test files:",
            len(self.files)
        )


    def __len__(self):

        return len(self.files)


    def __getitem__(self, index):

        filepath = self.files[index]

        label = self.labels[index]

        mel = extract_melspectrogram(
            filepath
        )

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel, label


# ============================================================
# CNN MODEL
# ============================================================

class CoughCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n==============================")
print("Loading Test Dataset")
print("==============================")


test_dataset = CoughTestDataset()


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD TRAINED CNN
# ============================================================

print("\n==============================")
print("Loading CNN Model")
print("==============================")


model = CoughCNN().to(DEVICE)


model.load_state_dict(
    torch.load(
        "model/tb_cnn.pth",
        map_location=DEVICE
    )
)


model.eval()


print("CNN model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_predictions = []


print("\n==============================")
print("Testing CNN")
print("==============================")


with torch.no_grad():

    for audio, labels in test_loader:

        audio = audio.to(DEVICE)

        outputs = model(audio)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n==============================")
print("CNN TEST RESULTS")
print("==============================")


print(
    f"\nTest Accuracy: {accuracy:.4f}"
)


print("\nClassification Report:")


print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Healthy",
            "TB"
        ]
    )
)


print("\nConfusion Matrix:")


print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)


print("\n==============================")
print("TESTING COMPLETE")
print("==============================")