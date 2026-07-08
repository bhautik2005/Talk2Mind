"""
train.py
========
End-to-end training entry point for Talk2Mind. Designed to be run cell-by-cell
in Google Colab (see README.md for the exact Colab steps), or as a script:

    python train.py --stage all

Stages
------
1. Fit the text TextVectorization layer on the training transcripts.
2. Train Model 1 (text), Model 2 (audio), Model 3 (video) independently,
   each on its own emotion-label target (multi-label sigmoid).
3. Freeze the three embedding extractors and train the Late Fusion ANN
   on the synchronized (text, audio, video) triples to predict the final
   Mental Well-Being Score.

Because dataset_loader.Talk2MindDataset yields
    ((text_tokens, audio_features, video_frames), labels)
per CSV row, all three modalities are already time/identity-synchronized -
every batch corresponds to the *same* set of utterances across modalities.
"""

import argparse
import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset_loader import Talk2MindDataset
from models_files.text_model import build_text_model, build_text_vectorizer
from models_files.audio_model import build_audio_model
from models_files.video_model import build_video_model
from models_files.fusion_model import build_fusion_model


def get_datasets():
    """Load train/val splits and fit the shared text vectorizer on train only."""
    train_ds = Talk2MindDataset("train")
    val_ds = Talk2MindDataset("val")

    vectorizer = build_text_vectorizer(train_ds.texts())
    train_ds.set_text_vectorizer(vectorizer)
    val_ds.set_text_vectorizer(vectorizer)
    return train_ds, val_ds, vectorizer


def _labels_only(ds_tf, modality_index):
    """Helper: reshape a full 3-modality tf.data.Dataset down to
    (single_modality_input, labels) so we can train one unimodal model at a time."""
    def _map(inputs, labels):
        return inputs[modality_index], labels
    return ds_tf.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def train_unimodal_models(train_ds, val_ds):
    callbacks = lambda name: [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.CHECKPOINT_DIR, f"{name}_best.keras"),
            save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    ]

    # --- Model 1: Text ---
    text_model_path = os.path.join(config.CHECKPOINT_DIR, "text_model_final.weights.h5")
    text_model, text_emb_model = build_text_model()
    loaded_text = False
    if os.path.exists(text_model_path):
        print("\n=== Loading existing Text Model weights ===")
        try:
            text_model.load_weights(text_model_path)
            loaded_text = True
            print("Successfully loaded Text Model weights.")
        except Exception as e:
            print(f"Warning: Could not load Text weights ({e}). Retraining...")
            
    if not loaded_text:
        print("\n=== Training Text Model ===")
        train_tf_text = train_ds.as_tf_dataset(shuffle=True, modalities=(0,))
        val_tf_text = val_ds.as_tf_dataset(shuffle=False, modalities=(0,))
        text_model.fit(_labels_only(train_tf_text, 0), validation_data=_labels_only(val_tf_text, 0),
                        epochs=config.EPOCHS_UNIMODAL, callbacks=callbacks("text_model"))
        text_model.save_weights(text_model_path)
        print("Saved Text model weights to", text_model_path)

    # --- Model 2: Audio ---
    audio_model_path = os.path.join(config.CHECKPOINT_DIR, "audio_model_final.weights.h5")
    audio_model, audio_emb_model = build_audio_model()
    loaded_audio = False
    if os.path.exists(audio_model_path):
        print("\n=== Loading existing Audio Model weights ===")
        try:
            audio_model.load_weights(audio_model_path)
            loaded_audio = True
            print("Successfully loaded Audio Model weights.")
        except Exception as e:
            print(f"Warning: Could not load Audio weights ({e}). Retraining...")
            
    if not loaded_audio:
        print("\n=== Training Audio (SER) Model ===")
        train_tf_audio = train_ds.as_tf_dataset(shuffle=True, modalities=(1,))
        val_tf_audio = val_ds.as_tf_dataset(shuffle=False, modalities=(1,))
        audio_model.fit(_labels_only(train_tf_audio, 1), validation_data=_labels_only(val_tf_audio, 1),
                         epochs=config.EPOCHS_UNIMODAL, callbacks=callbacks("audio_model"))
        audio_model.save_weights(audio_model_path)
        print("Saved Audio model weights to", audio_model_path)

    # --- Model 3: Video ---
    video_model_path = os.path.join(config.CHECKPOINT_DIR, "video_model_final.weights.h5")
    video_model, video_emb_model = build_video_model(backbone="custom")
    loaded_video = False
    if os.path.exists(video_model_path):
        print("\n=== Loading existing Video Model weights ===")
        try:
            video_model.load_weights(video_model_path)
            loaded_video = True
            print("Successfully loaded Video Model weights.")
        except Exception as e:
            print(f"Warning: Could not load Video weights ({e}). Retraining...")

    if not loaded_video:
        print("\n=== Training Video (FER) Model ===")
        train_tf_video = train_ds.as_tf_dataset(shuffle=True, modalities=(2,))
        val_tf_video = val_ds.as_tf_dataset(shuffle=False, modalities=(2,))
        video_model.fit(_labels_only(train_tf_video, 2), validation_data=_labels_only(val_tf_video, 2),
                         epochs=config.EPOCHS_UNIMODAL, callbacks=callbacks("video_model"))
        video_model.save_weights(video_model_path)
        print("Saved Video model weights to", video_model_path)


    return (text_model, text_emb_model), (audio_model, audio_emb_model), (video_model, video_emb_model)


def train_fusion_model(train_ds, val_ds, text_emb_model, audio_emb_model, video_emb_model):
    train_tf = train_ds.as_tf_dataset(shuffle=True)
    val_tf = val_ds.as_tf_dataset(shuffle=False)

    # NOTE: fusion labels here reuse the emotion columns as a stand-in target.
    # In production, replace with your own aggregated Mental Well-Being Score
    # (e.g. derived from the questionnaire) and switch config.FUSION_TASK
    # to "regression" with a single continuous column.
    fusion_model = build_fusion_model(
        text_emb_model, audio_emb_model, video_emb_model,
        freeze_unimodal=True, task=config.FUSION_TASK,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.CHECKPOINT_DIR, "fusion_model_best.keras"),
            save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    print("\n=== Training Late Fusion Model ===")
    fusion_model.fit(train_tf, validation_data=val_tf,
                      epochs=config.EPOCHS_FUSION, callbacks=callbacks)
    return fusion_model


def main(stage="all"):
    train_ds, val_ds, vectorizer = get_datasets()

    if stage in ("unimodal", "all"):
        (text_model, text_emb_model), (audio_model, audio_emb_model), \
            (video_model, video_emb_model) = train_unimodal_models(train_ds, val_ds)
    else:
        raise NotImplementedError(
            "For 'fusion'-only stage, load saved unimodal models/checkpoints "
            "first and pass their embedding sub-models into train_fusion_model()."
        )

    if stage in ("fusion", "all"):
        fusion_model = train_fusion_model(
            train_ds, val_ds, text_emb_model, audio_emb_model, video_emb_model
        )
        fusion_model.save(os.path.join(config.CHECKPOINT_DIR, "talk2mind_fusion_final.keras"))
        print("Saved final fusion model to", config.CHECKPOINT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["unimodal", "fusion", "all"], default="all")
    args = parser.parse_args()
    main(args.stage)
