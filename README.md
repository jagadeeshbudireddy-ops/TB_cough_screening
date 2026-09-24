# AI-Based TB Cough Screening

A Flask web application that accepts a WAV cough recording, converts the audio into a normalized mel spectrogram, and uses a PyTorch convolutional neural network to classify the recording as either a healthy-associated or TB-associated cough pattern.

> **Medical disclaimer:** This is a research and academic prototype. It is not a medical diagnostic device and must not be used to confirm or rule out tuberculosis. Always consult a qualified healthcare professional and use appropriate clinical testing.

## Features

- WAV cough-audio upload through a browser interface.
- Four-second audio normalization:
  - 16 kHz sample rate.
  - Mono audio.
  - Zero-padding for short files.
  - Truncation for longer files.
- 64-band mel-spectrogram extraction.
- Per-spectrogram standardization before inference.
- PyTorch CNN inference on CPU or CUDA GPU when available.
- Predicted class and model confidence display.
- English, Hindi, and Telugu interface translations.
- Light mode and dark mode.
- Theme and language preferences saved in the browser.
- Temporary uploaded audio is deleted after processing.
- Invalid, empty, missing, and non-WAV uploads are rejected.

## Project Structure

```text
TB_COUGH_SCREENING/
|
|-- app.py                         # Flask application and live inference route
|-- check_audio.py                 # MFCC feature-extraction helper script
|-- predict.py                     # Prediction placeholder; currently empty
|-- requirements.txt               # Python dependency list; currently empty
|
|-- model/
|   |-- tb_cnn.pth                 # Active PyTorch CNN weights used by app.py
|   |-- tb_model.pkl               # Legacy scikit-learn model artifact, if retained
|   `-- scaler.pkl                 # Legacy feature scaler artifact, if retained
|
|-- preprocessing/
|   `-- audio_features.py          # MFCC feature extraction helper
|
|-- static/
|   |-- background.png             # Page background image
|   `-- style.css                  # Page layout, colors, themes, and controls
|
|-- templates/
|   `-- index.html                 # Upload page, translations, and theme controls
|
|-- dataset/
|   |-- TB/                        # Local dataset category, if populated
|   `-- Non_TB/                    # Local dataset category, if populated
|
`-- uploads/                       # Temporary uploaded WAV files; normally empty
```

`__pycache__` folders and `.pyc` files are generated automatically by Python and are not source files.

## Runtime Flow

1. The browser sends a WAV file to `POST /`.
2. `app.py` checks that the file exists, has a filename, and ends in `.wav`.
3. The file is saved temporarily in `uploads/`.
4. `extract_mel_spectrogram()` loads the audio with `librosa` at 16 kHz and converts it to mono.
5. The audio is padded or cropped to exactly four seconds.
6. A 64-mel-bin spectrogram is generated using an FFT size of 512 and hop length of 256.
7. The spectrogram is converted to decibels and standardized using its mean and standard deviation.
8. The tensor is passed to the CNN on the selected device.
9. Softmax probabilities are calculated. The highest probability becomes the prediction confidence.
10. Class `1` is shown as `TB-associated cough pattern detected`; class `0` is shown as `Healthy-associated cough pattern detected`.
11. The temporary uploaded file is removed in the `finally` block.
12. The result page displays the translated result text and confidence.

## Model Architecture

The active model is `CoughCNN` in `app.py` and `train_cnn.py`:

```text
Input: 1 x mel-spectrogram
  -> Conv2d(1, 16, kernel_size=3, padding=1)
  -> ReLU
  -> MaxPool2d(2)
  -> Conv2d(16, 32, kernel_size=3, padding=1)
  -> ReLU
  -> MaxPool2d(2)
  -> Conv2d(32, 64, kernel_size=3, padding=1)
  -> ReLU
  -> AdaptiveAvgPool2d(1, 1)
  -> Flatten
  -> Dropout(0.4)
  -> Linear(64, 2)
```

The saved weights are loaded from `model/tb_cnn.pth`. The model architecture in `app.py` must remain compatible with the architecture used to create that file.

## Requirements

The current `requirements.txt` is empty, so install the packages required by the application manually or add pinned versions to that file.

The application imports:

- Flask
- librosa
- NumPy
- PyTorch
- PyTorch neural-network utilities

A typical installation is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install flask librosa numpy torch
```

If PowerShell blocks script activation, run Python directly from the environment or use the equivalent activation command for your shell.

### Verify Installation

```powershell
python -c "import flask, librosa, numpy, torch; print('Dependencies imported successfully')"
```

For a CUDA-enabled PyTorch installation, use the install command recommended for the CUDA version installed on your machine. CPU inference is supported automatically.

## Run the Web Application

From the project root:

```powershell
python app.py
```

The application starts Flask in debug mode. Open the URL printed by Flask, commonly:

```text
http://127.0.0.1:5000/
```

For a non-debug deployment, run Flask behind a production WSGI server and set an appropriate secret, host, and environment configuration. The current code is intended for local development and academic demonstration.

## Using the Interface

1. Open the application in a browser.
2. Choose a language from the selector:
   - English
   - Hindi
   - Telugu
3. Use the theme button to switch between light and dark mode.
4. Select a `.wav` cough recording.
5. Click **Analyze Cough**.
6. Read the predicted pattern and confidence percentage.

The selected language and theme are stored in browser `localStorage`. Clearing site data resets both preferences.

## Training the CNN

The training implementation is in `train_cnn.py`. It uses:

- Sample rate: `16000`
- Audio duration: `4` seconds
- Mel bins: `64`
- Batch size: `16`
- Epochs: `20`
- Learning rate: `0.001`
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Device: CUDA when available, otherwise CPU

The script expects a dataset arranged like this:

```text
Kaggle_RespiScan_Dataset/
`-- audio_wav/
    |-- train/
    |   |-- Healthy/
    |   `-- TB/
    `-- val/
        |-- Healthy/
        `-- TB/
```

Before training, update `DATASET_PATH` in `train_cnn.py`. The current source contains a machine-specific Windows path, so it will not work on another computer until changed.

Run training with:

```powershell
python train_cnn.py
```

Only the best validation model is saved, based on validation accuracy:

```text
model/tb_cnn.pth
```

The model used by the web app must be trained with the same preprocessing and compatible CNN architecture.

## Dataset Notes

The repository contains `dataset/TB` and `dataset/Non_TB` directories, but the current CNN training script reads from the external path configured by `DATASET_PATH`, specifically the `audio_wav/train` and `audio_wav/val` structure shown above. Keeping a local dataset folder does not automatically make it the training source.

Do not commit private, patient-identifiable, or restricted audio recordings to a public repository. Confirm the dataset license and consent requirements before redistribution.

## Audio Preprocessing Details

The active inference preprocessing is implemented directly in `app.py`:

```text
librosa.load(..., sr=16000, mono=True)
-> pad or crop to 64,000 samples
-> librosa.feature.melspectrogram(n_mels=64, n_fft=512, hop_length=256)
-> librosa.power_to_db(ref=np.max)
-> (mel - mean) / (std + 1e-8)
-> tensor shape [1, 1, 64, time]
```

`preprocessing/audio_features.py` and `check_audio.py` contain MFCC-based utilities. They are separate from the active mel-spectrogram pipeline used by `app.py` and should not be mixed with the CNN model unless the model is retrained for MFCC inputs.

## Error Handling

The application reports errors when:

- No `audio` field is submitted.
- No file is selected.
- The filename does not end with `.wav`.
- The audio file is empty or cannot be decoded.
- Feature extraction or model inference fails.

Uploaded files are removed after processing, including when an exception occurs. For production use, also add secure filename handling, upload size limits, authentication, rate limiting, and stronger file-content validation.

## Important Security Improvements for Deployment

The current application is suitable for local demonstration, but a public deployment should additionally:

- Replace `file.filename` with a sanitized filename using Werkzeug `secure_filename`.
- Avoid running Flask with `debug=True` in production.
- Configure a production WSGI server such as Waitress or Gunicorn where supported.
- Add maximum request and upload sizes.
- Validate file content, not only the filename extension.
- Store temporary uploads outside publicly served directories when appropriate.
- Add authentication and rate limiting if the service is exposed publicly.
- Log failures without exposing sensitive audio paths or user information.
- Keep medical results clearly labeled as research-only screening output.

## Troubleshooting

### `ModuleNotFoundError`

Install the missing dependency in the active environment:

```powershell
python -m pip install <package-name>
```

### Model file cannot be loaded

Confirm that this file exists relative to the project root:

```text
model/tb_cnn.pth
```

Run `python app.py` from the `TB_COUGH_SCREENING` directory so relative paths resolve correctly.

### Training dataset cannot be found

Update `DATASET_PATH` in `train_cnn.py` to the actual location of the dataset and confirm the `audio_wav/train` and `audio_wav/val` folders exist.

### Browser does not show the latest CSS or JavaScript

Refresh the page with a hard reload and check the browser developer console for JavaScript errors. The interface stores language and theme settings in `localStorage`, so clearing site data can reset the UI state.

### CUDA errors

The application automatically falls back to CPU when CUDA is unavailable. If CUDA is installed but failing, install a PyTorch build compatible with the installed driver, or temporarily force CPU by changing the device selection in `app.py` and `train_cnn.py`.

## Development Checklist

Before sharing a change:

- Start the app with `python app.py`.
- Test a valid WAV file.
- Test a non-WAV file.
- Test an empty or invalid audio file.
- Check light mode and dark mode.
- Check English, Hindi, and Telugu.
- Confirm the result and confidence labels change with the selected language.
- Confirm temporary uploaded files are removed.
- Confirm `model/tb_cnn.pth` loads successfully.

## License and Data Usage

No license file is currently included in the project. Add an appropriate software license before distributing the source. Treat all audio data according to its original dataset license, privacy requirements, and applicable research or medical regulations.
