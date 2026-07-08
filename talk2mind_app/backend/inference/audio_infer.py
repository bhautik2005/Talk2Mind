"""
audio_infer.py
==============
Loads audio_model_final.h5 (once, at import time) and exposes predict_audio().
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_paths as mp
from preprocessing.audio_preprocessing import extract_features

_model = None


def _load():
    global _model
    if _model is None:
        print(f"[audio_infer] trying model candidates: {mp.AUDIO_MODEL_CANDIDATES}")
        loaded_model = None
        for candidate in mp.AUDIO_MODEL_CANDIDATES:
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                m = tf.keras.models.load_model(candidate, compile=False)
                expected_shape = m.input_shape
                expected_steps = expected_shape[1]
                expected_features = expected_shape[2]
                current_steps = mp.AUDIO_TIME_STEPS
                current_features = mp.AUDIO_N_MFCC + mp.AUDIO_N_MELS + 2
                
                if expected_steps == current_steps and expected_features == current_features:
                    _model = m
                    print(f"[audio_infer] loaded matching model from {candidate} (shape {expected_shape})")
                    return
                else:
                    print(f"[audio_infer] candidate {candidate} has shape {expected_shape}, current config is ({current_steps}, {current_features})")
                    if loaded_model is None:
                        loaded_model = m
            except Exception as e:
                print(f"[audio_infer] failed to load candidate {candidate}: {e}")

        if loaded_model is not None:
            _model = loaded_model
            print(f"[audio_infer] Warning: using model with mismatched shape {_model.input_shape}")
            return

        raise RuntimeError("Could not load any compatible audio model.")


def predict_audio(wav_path: str):
    """
    Parameters
    ----------
    wav_path : path to a recorded/uploaded .wav clip

    Returns
    -------
    dict: {label: score, ...} for each entry in model_paths.EMOTION_LABELS
    """
    _load()
    
    # Inspect model input shape dynamically
    shape = _model.input_shape
    target_time_steps = shape[1] if (len(shape) > 1 and shape[1] is not None) else mp.AUDIO_TIME_STEPS
    target_features = shape[2] if (len(shape) > 2 and shape[2] is not None) else (mp.AUDIO_N_MFCC + mp.AUDIO_N_MELS + 2)
    
    # Adapt parameters based on target shape
    n_mfcc, n_mels, duration = mp.AUDIO_N_MFCC, mp.AUDIO_N_MELS, mp.AUDIO_DURATION_SEC
    if target_features == 81:
        n_mfcc = 13
        n_mels = 66
    if target_time_steps == 94:
        duration = 3.0
        
    features = extract_features(
        wav_path,
        time_steps=target_time_steps,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        duration=duration
    )  # (target_time_steps, target_features)
    
    features = np.expand_dims(features, axis=0)         # (1, target_time_steps, target_features)
    preds = _model.predict(features, verbose=0)[0]

    scores = {label: float(preds[i]) for i, label in enumerate(mp.EMOTION_LABELS)}
    return scores