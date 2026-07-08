"""
audio_model.py
==============
MODEL 2: Speech Emotion Recognition (SER).

Input: (time_steps, feature_dim) MFCC + log-mel + prosody feature map
       produced by preprocessing/audio_preprocessing.py

Architecture:
    1D-CNN blocks (acoustic pattern extraction along time)
    -> LSTM (temporal dynamics of voice over the utterance)
    -> ANN head -> emotion classification

Exposes:
    build_audio_model() -> (full_model, embedding_model)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.audio_preprocessing import audio_feature_dim


def build_audio_model(time_steps=config.AUDIO_TIME_STEPS,
                       feature_dim=None,
                       embedding_out_dim=config.AUDIO_EMBEDDING_OUT_DIM,
                       num_classes=config.NUM_CLASSES):
    """
    Returns
    -------
    full_model : tf.keras.Model
        input: (batch, time_steps, feature_dim)
        output: sigmoid multi-label emotion scores (batch, num_classes)
    embedding_model : tf.keras.Model
        Same input -> 'speech_emotion_embedding' (batch, embedding_out_dim)
    """
    feature_dim = feature_dim or audio_feature_dim()
    inputs = layers.Input(shape=(time_steps, feature_dim), name="audio_features")

    # --- 1D CNN block: local acoustic pattern extraction across time ---
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # --- LSTM block: temporal fluctuation of voice over the clip ---
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)

    # --- ANN head ---
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    embedding = layers.Dense(embedding_out_dim, activation="relu",
                              name="speech_emotion_embedding")(x)

    x = layers.Dropout(0.3)(embedding)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="audio_output")(x)

    full_model = Model(inputs, outputs, name="SpeechEmotionModel")
    embedding_model = Model(inputs, embedding, name="AudioEmbeddingExtractor")

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return full_model, embedding_model


if __name__ == "__main__":
    model, emb_model = build_audio_model()
    model.summary()
