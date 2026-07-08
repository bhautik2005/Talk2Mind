"""
config.py
=========
Central configuration for Talk2Mind.

Edit ROOT_DIR to point at your Google Drive mount in Colab, e.g.:
    ROOT_DIR = "/content/drive/MyDrive/CMU-MOSEI"

This matches the folder structure you showed:

CMU-MOSEI/
├── Audio_chunk/
│   ├── Train_modified/        <- audio clips (.wav) for training
│   ├── Val_modified/
│   ├── Test_modified/
│   ├── Train_original/ ...
│   ├── Val_original/ ...
│   └── Labels/
│       ├── Data_Train_modified.csv
│       ├── Data_Val_modified.csv
│       ├── Data_Test_modified.csv
│       ├── Data_Val_original_without_neg_time.csv
│       └── Data_Test_original_without_neg_time.csv
└── Video_Dataset/
    ├── Train/ Val/ Test/      <- (mirrors Audio_chunk split naming)

Adjust SPLIT_FOLDER_MAP / CSV names below if your folder names differ slightly.
"""

import os

# ------------------------------------------------------------------
# 1. ROOT PATHS  (edit this for your Colab / local environment)
# ------------------------------------------------------------------
# ROOT_DIR = "/content/drive/MyDrive/CMU-MOSEI"          # <-- CHANGE ME IN COLAB
# Use os.path.join to avoid accidental escape-sequence issues on Windows.
ROOT_DIR = os.path.join("DataSet", "CMU-MOSEI")          # <-- CHANGE ME IN COLAB

AUDIO_ROOT = os.path.join(ROOT_DIR, "Audio_chunk")
VIDEO_ROOT = os.path.join(ROOT_DIR, "Video_Dataset")
# Labels directory may live under Audio_chunk/Labels (original layout)
# or directly under CMU-MOSEI/Labels depending on how the dataset was unpacked.
if os.path.isdir(os.path.join(AUDIO_ROOT, "Labels")):
    LABELS_DIR = os.path.join(AUDIO_ROOT, "Labels")
elif os.path.isdir(os.path.join(ROOT_DIR, "Labels")):
    LABELS_DIR = os.path.join(ROOT_DIR, "Labels")
else:
    # Fallback to the audio labels path (will raise FileNotFound later if absent).
    LABELS_DIR = os.path.join(AUDIO_ROOT, "Labels")

# Which CSV (label file) and which audio/video subfolder belong to each split.
# "_modified" versions are the ones with the negative-time issue already fixed
# and are what we train on by default.
SPLIT_CSV = {
    "train": os.path.join(LABELS_DIR, "Data_Train_modified.csv"),
    "val":   os.path.join(LABELS_DIR, "Data_Val_modified.csv"),
    "test":  os.path.join(LABELS_DIR, "Data_Test_modified.csv"),
}

SPLIT_AUDIO_DIR = {
    "train": os.path.join(AUDIO_ROOT, "Train_modified"),
    "val":   os.path.join(AUDIO_ROOT, "Val_modified"),
    "test":  os.path.join(AUDIO_ROOT, "Test_modified"),
}

SPLIT_VIDEO_DIR = {
    "train": os.path.join(VIDEO_ROOT, "Train"),
    "val":   os.path.join(VIDEO_ROOT, "Val"),
    "test":  os.path.join(VIDEO_ROOT, "Test"),
}

# ------------------------------------------------------------------
# 2. CSV COLUMN NAMES  (rename these to match your actual CSV headers)
# ------------------------------------------------------------------
# CMU-MOSEI style label CSVs typically carry a video_id + clip/segment id
# (used to build the audio/video filename), the transcript text, and either
# a continuous sentiment score and/or discrete emotion columns.
COL_VIDEO_ID = "video"
COL_START_TIME = "start_time"
COL_END_TIME = "end_time"
COL_TEXT = "text"

# Continuous sentiment label (regression target), if present.
COL_SENTIMENT = "sentiment"

# Discrete emotion columns (CMU-MOSEI: happy, sad, anger, surprise, disgust, fear).
EMOTION_COLUMNS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

# Filename pattern used to locate the matching audio/video file for a row.
# Adjust to match what you actually see on disk, e.g. "{video_id}_{start_time:.4f}_{end_time:.4f}.wav"
AUDIO_FILENAME_TEMPLATE = "{video_id}_{start_time:.4f}_{end_time:.4f}.wav"
VIDEO_FILENAME_TEMPLATE = "{video_id}_{start_time:.4f}_{end_time:.4f}.mp4"

# ------------------------------------------------------------------
# 3. TEXT MODEL HYPERPARAMETERS
# ------------------------------------------------------------------
TEXT_VOCAB_SIZE = 20000
TEXT_MAX_LEN = 50
TEXT_EMBED_DIM = 128
TEXT_LSTM_UNITS = 128
TEXT_EMBEDDING_OUT_DIM = 128   # size of the extracted text_emotion_embedding

# ------------------------------------------------------------------
# 4. AUDIO MODEL HYPERPARAMETERS
# ------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 4.0                      # pad/truncate every clip to this length
AUDIO_N_MFCC = 40
AUDIO_N_MELS = 64
AUDIO_HOP_LENGTH = 512
AUDIO_N_FFT = 1024
AUDIO_TIME_STEPS = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SEC / AUDIO_HOP_LENGTH) + 1
AUDIO_EMBEDDING_OUT_DIM = 128

# ------------------------------------------------------------------
# 5. VIDEO MODEL HYPERPARAMETERS
# ------------------------------------------------------------------
VIDEO_NUM_FRAMES = 16          # frames sampled per clip
VIDEO_FRAME_SIZE = 96          # frame resized to (96,96,3) before CNN
VIDEO_EMBEDDING_OUT_DIM = 128

# ------------------------------------------------------------------
# 6. FUSION / TRAINING
# ------------------------------------------------------------------
NUM_CLASSES = len(EMOTION_COLUMNS)   # multi-label emotion classification
FUSION_TASK = "regression"           # "regression" -> single well-being score
                                      # "classification" -> multi-class risk level
BATCH_SIZE = 8
EPOCHS_UNIMODAL = 10
EPOCHS_FUSION = 25
LEARNING_RATE = 1e-4

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
