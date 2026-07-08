"""
fusion_model.py
================
MULTIMODAL FUSION LAYER (Late Fusion ANN)

Takes the three per-modality embedding extractors (text, speech, visual),
concatenates their embeddings, and passes the fused vector through a Dense
ANN (with Dropout + BatchNorm) to produce the final Mental Well-Being Score.

Two usage modes:
  1. build_fusion_model(...)      -> raw multimodal inputs -> score
                                      (embeddings computed internally,
                                      end-to-end differentiable / trainable)
  2. build_fusion_head_only(...)  -> pre-computed embeddings -> score
                                      (use this if you trained each unimodal
                                      model separately and cached embeddings)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from models_files.text_model import build_text_model
from models_files.audio_model import build_audio_model
from models_files.video_model import build_video_model


def _fusion_head(fused_vector, task=config.FUSION_TASK, num_classes=config.NUM_CLASSES):
    """Shared Dense ANN block used by both fusion entry points below."""
    x = layers.Dense(256, activation="relu")(fused_vector)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    if task == "regression":
        # Single continuous Mental Well-Being Score, e.g. scaled to 0-100.
        outputs = layers.Dense(1, activation="linear", name="wellbeing_score")(x)
    else:
        # Discrete risk-level classification (e.g. low/moderate/high risk).
        outputs = layers.Dense(num_classes, activation="softmax", name="wellbeing_class")(x)
    return outputs


def build_fusion_model(text_embedding_model, audio_embedding_model, video_embedding_model,
                        freeze_unimodal=True, task=config.FUSION_TASK):
    """
    End-to-end fusion model: takes the three RAW modality inputs, runs each
    through its embedding extractor, concatenates, and predicts the score.

    Parameters
    ----------
    text_embedding_model, audio_embedding_model, video_embedding_model :
        The `embedding_model` returned by build_text_model / build_audio_model
        / build_video_model (already trained, ideally).
    freeze_unimodal : bool
        If True, the unimodal backbones are frozen and only the fusion head
        is trained (recommended: train unimodal models first, then fuse).
        If False, everything is fine-tuned jointly end-to-end.
    """
    if freeze_unimodal:
        text_embedding_model.trainable = False
        audio_embedding_model.trainable = False
        video_embedding_model.trainable = False

    text_in = text_embedding_model.input
    audio_in = audio_embedding_model.input
    video_in = video_embedding_model.input

    text_emb = text_embedding_model(text_in)
    audio_emb = audio_embedding_model(audio_in)
    video_emb = video_embedding_model(video_in)

    fused = layers.Concatenate(name="multimodal_fusion")([text_emb, audio_emb, video_emb])
    outputs = _fusion_head(fused, task=task)

    model = Model(inputs=[text_in, audio_in, video_in], outputs=outputs, name="Talk2MindFusionModel")

    loss = "mse" if task == "regression" else "categorical_crossentropy"
    metrics = ["mae"] if task == "regression" else ["accuracy"]
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss=loss, metrics=metrics)
    return model


def build_fusion_head_only(text_dim=config.TEXT_EMBEDDING_OUT_DIM,
                            audio_dim=config.AUDIO_EMBEDDING_OUT_DIM,
                            video_dim=config.VIDEO_EMBEDDING_OUT_DIM,
                            task=config.FUSION_TASK):
    """
    Lightweight fusion model that takes ALREADY-EXTRACTED embedding vectors
    as input (e.g. cached to .npy after running the unimodal models once).
    Much faster to train/iterate on than the end-to-end version above.
    """
    text_in = layers.Input(shape=(text_dim,), name="text_emotion_embedding")
    audio_in = layers.Input(shape=(audio_dim,), name="speech_emotion_embedding")
    video_in = layers.Input(shape=(video_dim,), name="visual_emotion_embedding")

    fused = layers.Concatenate(name="multimodal_fusion")([text_in, audio_in, video_in])
    outputs = _fusion_head(fused, task=task)

    model = Model(inputs=[text_in, audio_in, video_in], outputs=outputs,
                  name="Talk2MindFusionHead")

    loss = "mse" if task == "regression" else "categorical_crossentropy"
    metrics = ["mae"] if task == "regression" else ["accuracy"]
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    _, text_emb = build_text_model()
    _, audio_emb = build_audio_model()
    _, video_emb = build_video_model()
    fusion = build_fusion_model(text_emb, audio_emb, video_emb)
    fusion.summary()
