"""
audio_preprocessing.py (inference copy)
========================================
Same feature extraction used at training time — MFCC + log-mel + pitch/energy,
fixed-length padded. Must stay in sync with whatever produced
audio_model_final.h5's training data, or predictions will be garbage.
"""

import numpy as np
import librosa
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_paths as mp


def load_and_fix_length(path, sr=mp.AUDIO_SAMPLE_RATE, duration=None):
    if duration is None:
        duration = mp.AUDIO_DURATION_SEC
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y


def extract_features(path, time_steps=None, n_mfcc=None, n_mels=None, duration=None):
    """Returns np.ndarray of shape (time_steps, n_mfcc + n_mels + 2)."""
    if time_steps is None:
        time_steps = mp.AUDIO_TIME_STEPS
    if n_mfcc is None:
        n_mfcc = mp.AUDIO_N_MFCC
    if n_mels is None:
        n_mels = mp.AUDIO_N_MELS
    if duration is None:
        duration = mp.AUDIO_DURATION_SEC

    y = load_and_fix_length(path, duration=duration)

    mfcc = librosa.feature.mfcc(
        y=y, sr=mp.AUDIO_SAMPLE_RATE, n_mfcc=n_mfcc,
        n_fft=mp.AUDIO_N_FFT, hop_length=mp.AUDIO_HOP_LENGTH,
    )
    mel = librosa.feature.melspectrogram(
        y=y, sr=mp.AUDIO_SAMPLE_RATE, n_mels=n_mels,
        n_fft=mp.AUDIO_N_FFT, hop_length=mp.AUDIO_HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel)
    rms = librosa.feature.rms(y=y, hop_length=mp.AUDIO_HOP_LENGTH)
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=mp.AUDIO_SAMPLE_RATE, hop_length=mp.AUDIO_HOP_LENGTH,
        )
        f0 = np.nan_to_num(f0)[np.newaxis, :]
    except Exception:
        f0 = np.zeros_like(rms)

    T = min(mfcc.shape[1], log_mel.shape[1], rms.shape[1], f0.shape[1])
    feat = np.concatenate([mfcc[:, :T], log_mel[:, :T], rms[:, :T], f0[:, :T]], axis=0).T

    if feat.shape[0] < time_steps:
        feat = np.concatenate([feat, np.zeros((time_steps - feat.shape[0], feat.shape[1]))], axis=0)
    else:
        feat = feat[:time_steps, :]

    mean, std = feat.mean(), feat.std() + 1e-6
    return ((feat - mean) / std).astype(np.float32)