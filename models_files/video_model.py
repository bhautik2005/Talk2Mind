"""
video_model.py
==============
MODEL 3: Facial Emotion Recognition (FER).

Input: (num_frames, H, W, 3) sequence of face crops produced by
       preprocessing/video_preprocessing.py

Architecture:
    TimeDistributed CNN backbone (custom, OR frozen EfficientNetB0 as in the
    Talk2Mind architecture diagram) -> per-frame spatial embeddings
    -> LSTM across frames (aggregate micro-expression dynamics over time)
    -> ANN head -> facial emotion classification

Exposes:
    build_video_model(backbone="custom"|"efficientnet") -> (full_model, embedding_model)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _custom_cnn_backbone(frame_size):
    """A lightweight custom CNN backbone applied per-frame via TimeDistributed."""
    inp = layers.Input(shape=(frame_size, frame_size, 3))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    return Model(inp, x, name="CustomCNNBackbone")


def _efficientnet_backbone(frame_size):
    """Frozen EfficientNetB0 backbone (transfer learning), as shown in the
    architecture diagram. Requires internet access to download ImageNet
    weights the first time it's run."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(frame_size, frame_size, 3), pooling="avg",
    )
    base.trainable = False  # freeze for feature extraction
    return base


def build_video_model(num_frames=config.VIDEO_NUM_FRAMES,
                       frame_size=config.VIDEO_FRAME_SIZE,
                       embedding_out_dim=config.VIDEO_EMBEDDING_OUT_DIM,
                       num_classes=config.NUM_CLASSES,
                       backbone="custom"):
    """
    Parameters
    ----------
    backbone : "custom" (fast, trains from scratch) or
               "efficientnet" (frozen transfer-learning backbone, per the
               architecture diagram)

    Returns
    -------
    full_model : tf.keras.Model
        input: (batch, num_frames, frame_size, frame_size, 3)
        output: sigmoid multi-label emotion scores (batch, num_classes)
    embedding_model : tf.keras.Model
        Same input -> 'visual_emotion_embedding' (batch, embedding_out_dim)
    """
    inputs = layers.Input(shape=(num_frames, frame_size, frame_size, 3), name="video_frames")

    if backbone == "efficientnet":
        cnn = _efficientnet_backbone(frame_size)
    else:
        cnn = _custom_cnn_backbone(frame_size)

    # Apply the CNN backbone to every frame independently, then aggregate temporally.
    x = layers.TimeDistributed(cnn, name="td_cnn_backbone")(inputs)  # (batch, frames, feat)

    # --- LSTM block: aggregate micro-expression dynamics across frames ---
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)

    # --- ANN head ---
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    embedding = layers.Dense(embedding_out_dim, activation="relu",
                              name="visual_emotion_embedding")(x)

    x = layers.Dropout(0.3)(embedding)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="video_output")(x)

    full_model = Model(inputs, outputs, name="FacialEmotionModel")
    embedding_model = Model(inputs, embedding, name="VideoEmbeddingExtractor")

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return full_model, embedding_model


if __name__ == "__main__":
    model, emb_model = build_video_model(backbone="custom")
    model.summary()
