"""
audio_preprocessing.py
=======================
Turns a raw .wav file into a fixed-size (time_steps, n_mfcc + n_mels) feature
map (MFCCs stacked with log-mel spectrogram) that the SER CNN+LSTM model
consumes.

Uses librosa. Every clip is padded/truncated to config.AUDIO_DURATION_SEC so
that all feature maps have identical shape.
"""

import numpy as np
import librosa

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_and_fix_length(path, sr=config.AUDIO_SAMPLE_RATE, duration=config.AUDIO_DURATION_SEC):
    """Load audio, resample, and pad/truncate to a fixed duration."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y


def extract_features(path):
    """
    Extract MFCCs + log-mel spectrogram from an audio file and stack them
    along the feature axis.

    Returns
    -------
    np.ndarray of shape (config.AUDIO_TIME_STEPS, n_mfcc + n_mels)
    """
    y = load_and_fix_length(path)

    mfcc = librosa.feature.mfcc(
        y=y, sr=config.AUDIO_SAMPLE_RATE, n_mfcc=config.AUDIO_N_MFCC,
        n_fft=config.AUDIO_N_FFT, hop_length=config.AUDIO_HOP_LENGTH,
    )  # (n_mfcc, T)

    mel = librosa.feature.melspectrogram(
        y=y, sr=config.AUDIO_SAMPLE_RATE, n_mels=config.AUDIO_N_MELS,
        n_fft=config.AUDIO_N_FFT, hop_length=config.AUDIO_HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel)  # (n_mels, T)

    # Extra prosodic scalars broadcast across time: pitch (f0) and RMS energy.
    rms = librosa.feature.rms(y=y, hop_length=config.AUDIO_HOP_LENGTH)  # (1, T)
    try:
        f0 = librosa.yin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=config.AUDIO_SAMPLE_RATE, hop_length=config.AUDIO_HOP_LENGTH,
        )
        f0 = np.nan_to_num(f0)[np.newaxis, :]
    except Exception:
        f0 = np.zeros_like(rms)

    # Make sure all feature blocks share the same time dimension (T).
    T = min(mfcc.shape[1], log_mel.shape[1], rms.shape[1], f0.shape[1])
    mfcc, log_mel, rms, f0 = mfcc[:, :T], log_mel[:, :T], rms[:, :T], f0[:, :T]

    feat = np.concatenate([mfcc, log_mel, rms, f0], axis=0)  # (n_mfcc+n_mels+2, T)
    feat = feat.T  # -> (T, features)  time-major for LSTM

    # Pad/truncate the time axis to a fixed length so batches stack cleanly.
    fixed_T = config.AUDIO_TIME_STEPS
    if feat.shape[0] < fixed_T:
        pad = np.zeros((fixed_T - feat.shape[0], feat.shape[1]))
        feat = np.concatenate([feat, pad], axis=0)
    else:
        feat = feat[:fixed_T, :]

    # Per-utterance normalization (z-score) for stable training.
    mean, std = feat.mean(), feat.std() + 1e-6
    feat = (feat - mean) / std
    return feat.astype(np.float32)


def audio_feature_dim():
    """Number of feature channels produced per time step (used to set model input shape)."""
    return config.AUDIO_N_MFCC + config.AUDIO_N_MELS + 2  # + rms + f0
