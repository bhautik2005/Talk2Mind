"""
text_infer.py
=============
Loads text_model_final.h5 (once, at import time) and exposes predict_text().
"""

import os
import pickle
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import model_paths as mp

try:
    from models_files.text_model import build_text_model
except Exception as exc:  # pragma: no cover - fallback import path guard
    build_text_model = None
    print(f"[text_infer] could not import build_text_model: {exc}")

_model = None
_vectorizer = None


def _load():
    global _model, _vectorizer
    if _model is None:
        print(f"[text_infer] trying model candidates: {mp.TEXT_MODEL_CANDIDATES}")
        loaded = False
        for candidate in mp.TEXT_MODEL_CANDIDATES:
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                if os.path.basename(candidate).endswith(".weights.h5"):
                    if build_text_model is None:
                        raise ImportError("build_text_model is unavailable")
                    model, _ = build_text_model()
                    model.load_weights(candidate)
                    _model = model
                    print(f"[text_infer] loaded weights from {candidate}")
                else:
                    _model = tf.keras.models.load_model(candidate)
                    print(f"[text_infer] loaded model from {candidate}")
                loaded = True
                break
            except Exception as exc:
                print(f"[text_infer] failed to load {candidate}: {exc}")

        if not loaded:
            raise RuntimeError(
                "Could not load any compatible text model. Expected one of: "
                + ", ".join(mp.TEXT_MODEL_CANDIDATES)
            )

    if _vectorizer is None:
        if os.path.exists(mp.TEXT_VOCAB_PATH):
            vocab = pickle.load(open(mp.TEXT_VOCAB_PATH, "rb"))
            _vectorizer = TextVectorization(
                max_tokens=len(vocab), output_mode="int",
                output_sequence_length=mp.TEXT_MAX_LEN,
            )
            _vectorizer.set_vocabulary(vocab)
            print(f"[text_infer] loaded vocabulary ({len(vocab)} tokens) from {mp.TEXT_VOCAB_PATH}")
        else:
            raise FileNotFoundError(
                f"Missing {mp.TEXT_VOCAB_PATH}. Export your training-time "
                f"TextVectorization vocabulary with "
                f"pickle.dump(vectorizer.get_vocabulary(), open('text_vectorizer_vocab.pkl','wb')) "
                f"and place it at that path (see model_paths.py)."
            )


def predict_text(text: str):
    """
    Parameters
    ----------
    text : the user's typed check-in response / questionnaire answer

    Returns
    -------
    dict: {label: score, ...} for each entry in model_paths.EMOTION_LABELS,
          plus a summary 'embedding-derived' distress score.
    """
    _load()
    tokens = _vectorizer(tf.constant([text]))
    preds = _model.predict(tokens, verbose=0)[0]  # shape (num_classes,)

    scores = {label: float(preds[i]) for i, label in enumerate(mp.EMOTION_LABELS)}
    return scores