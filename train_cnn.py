import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\jagad\Downloads\archive\Kaggle_RespiScan_Dataset"

SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 64

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

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
# DATASET
# ============================================================

class CoughDataset(Dataset):

    def __init__(self, split):

        self.files = []
        self.labels = []

        for label, class_name in enumerate(
            ["Healthy", "TB"]
        ):

            folder = os.path.join(
                DATASET_PATH,
                "audio_wav",
                split,
                class_name
            )

            print(
                f"\nLoading {class_name} files from {split}..."
            )

            for filename in os.listdir(folder):

                if not filename.lower().endswith(".wav"):
                    continue

                filepath = os.path.join(
                    folder,
                    filename
                )

                # ------------------------------------------------
                # ACTUALLY TEST THE AUDIO FILE
                # ------------------------------------------------

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

                    print(
                        "Reason:",
                        str(e)
                    )

                    continue

                # Valid file
                self.files.append(filepath)
                self.labels.append(label)

        print(
            f"{split} dataset: {len(self.files)}"
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
# LOAD DATA
# ============================================================

print("\n==============================")
print("Loading datasets")
print("==============================")


train_dataset = CoughDataset("train")

val_dataset = CoughDataset("val")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CREATE MODEL
# ============================================================

model = CoughCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n==============================")
print("Training CNN")
print("==============================")


best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_correct = 0
    train_total = 0

    train_loss_total = 0.0


    for audio, labels in train_loader:

        audio = audio.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(audio)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_correct += (
            predictions == labels
        ).sum().item()


        train_total += labels.size(0)

        train_loss_total += loss.item()


    train_accuracy = (
        train_correct /
        train_total
    )

    train_loss = (
        train_loss_total /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0

    val_loss_total = 0.0


    with torch.no_grad():

        for audio, labels in val_loader:

            audio = audio.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(audio)

            loss = criterion(
                outputs,
                labels
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += labels.size(0)

            val_loss_total += loss.item()


    val_accuracy = (
        val_correct /
        val_total
    )

    val_loss = (
        val_loss_total /
        len(val_loader)
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        os.makedirs(
            "model",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            "model/tb_cnn.pth"
        )

        print(
            "  -> Best CNN model saved!"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================")
print("CNN TRAINING COMPLETE")
print("==============================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Model saved at:"
)

print(
    "model/tb_cnn.pth"
)