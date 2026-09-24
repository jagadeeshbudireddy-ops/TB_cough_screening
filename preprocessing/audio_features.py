import librosa
import numpy as np


def extract_features(audio_path):
    """
    Extract MFCC features from a cough audio file.
    """

    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Skip empty audio files
    if audio is None or len(audio) == 0:
        raise ValueError("Empty audio file")

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40,
        n_fft=512
    )

    # Convert variable-length audio into fixed-size features
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return features
