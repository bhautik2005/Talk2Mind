"""
text_model.py
=============
MODEL 1: Chat/Text model.

Embedding -> Bidirectional LSTM (sequential linguistic context)
          -> Dense ANN head (classification / risk indicator)

Exposes:
    build_text_model()          -> (full_model, embedding_model)
    TextEmbeddingExtractor      -> thin wrapper class used by the fusion script
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_text_model(vocab_size=config.TEXT_VOCAB_SIZE,
                      max_len=config.TEXT_MAX_LEN,
                      embed_dim=config.TEXT_EMBED_DIM,
                      lstm_units=config.TEXT_LSTM_UNITS,
                      embedding_out_dim=config.TEXT_EMBEDDING_OUT_DIM,
                      num_classes=config.NUM_CLASSES):
    """
    Build the text model.

    Returns
    -------
    full_model : tf.keras.Model
        input: token ids (batch, max_len)
        output: sigmoid multi-label emotion/risk scores (batch, num_classes)
    embedding_model : tf.keras.Model
        Same input, but outputs the penultimate dense layer
        ('text_emotion_embedding') of shape (batch, embedding_out_dim).
        Shares weights with full_model (same layer objects).
    """
    inputs = layers.Input(shape=(max_len,), dtype="int32", name="text_tokens")

    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                          mask_zero=True, name="token_embedding")(inputs)

    # RNN block: stacked BiLSTM to capture sequential linguistic context.
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True), name="bilstm_1"
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2), name="bilstm_2"
    )(x)

    # ANN head.
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    embedding = layers.Dense(embedding_out_dim, activation="relu",
                              name="text_emotion_embedding")(x)

    x = layers.Dropout(0.3)(embedding)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="text_output")(x)

    full_model = Model(inputs, outputs, name="TextEmotionModel")
    embedding_model = Model(inputs, embedding, name="TextEmbeddingExtractor")

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return full_model, embedding_model


def build_text_vectorizer(train_texts, vocab_size=config.TEXT_VOCAB_SIZE,
                           max_len=config.TEXT_MAX_LEN):
    """
    Fit a TextVectorization layer on the training transcripts. Use this to
    turn raw strings into the integer token sequences the model expects.
    """
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_sequence_length=max_len
    )
    vectorizer.adapt(train_texts)
    return vectorizer


if __name__ == "__main__":
    model, emb_model = build_text_model()
    model.summary()
