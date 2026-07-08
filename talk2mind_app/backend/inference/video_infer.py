"""
video_infer.py
==============
Loads video_model_final.h5 (once, at import time) and exposes predict_video().
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_paths as mp
from preprocessing.video_preprocessing import extract_face_sequence

_model = None


def _load():
    global _model
    if _model is None:
        print(f"[video_infer] trying model candidates: {mp.VIDEO_MODEL_CANDIDATES}")
        loaded = False
        for candidate in mp.VIDEO_MODEL_CANDIDATES:
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                _model = tf.keras.models.load_model(candidate, compile=False)
                print(f"[video_infer] loaded model from {candidate}")
                loaded = True
                break
            except Exception as e:
                print(f"[video_infer] failed to load candidate {candidate}: {e}")

        if not loaded:
            raise RuntimeError(
                "Could not load any compatible video model. Expected one of: "
                + ", ".join(mp.VIDEO_MODEL_CANDIDATES)
            )


def predict_video(video_path: str):
    """
    Parameters
    ----------
    video_path : path to a short recorded/uploaded video clip (webcam or upload)

    Returns
    -------
    dict: {label: score, ...} for each entry in model_paths.EMOTION_LABELS
    """
    _load()
    
    # Inspect model input shape dynamically
    shape = _model.input_shape
    target_num_frames = shape[1] if (len(shape) > 1 and shape[1] is not None) else mp.VIDEO_NUM_FRAMES
    target_frame_size = shape[2] if (len(shape) > 2 and shape[2] is not None) else mp.VIDEO_FRAME_SIZE
    
    frames = extract_face_sequence(video_path, num_frames=target_num_frames, frame_size=target_frame_size)          # (num_frames, H, W, 3)
    frames = np.expand_dims(frames, axis=0)              # (1, num_frames, H, W, 3)
    preds = _model.predict(frames, verbose=0)[0]

    scores = {label: float(preds[i]) for i, label in enumerate(mp.EMOTION_LABELS)}
    return scores