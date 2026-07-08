"""
model_paths.py
==============
*** THIS IS THE ONLY FILE YOU NEED TO EDIT TO POINT AT YOUR TRAINED MODELS ***

Put your 3 .h5 files anywhere (the `models/` folder next to this project is
the default), then update the 3 paths below if you place them elsewhere.
"""

import os

# Folder containing this backend/ directory (i.e. the app project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

# ------------------------------------------------------------------
# >>>>>>>>>>>>>>>>>>>>  EDIT THESE 4 PATHS  <<<<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------------
TEXT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "text_model_final.h5")
TEXT_MODEL_CANDIDATES = [
    TEXT_MODEL_PATH,
    os.path.join(REPO_ROOT, "models", "text_model_best.keras"),
    os.path.join(REPO_ROOT, "models", "text_model_final.weights.h5"),
]
AUDIO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "audio_model_final.h5")
AUDIO_MODEL_CANDIDATES = [
    AUDIO_MODEL_PATH,
    os.path.join(PROJECT_ROOT, "models", "best_audio_model.keras"),
    os.path.join(REPO_ROOT, "models", "audio_model_best.keras"),
]
VIDEO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "video_model_final.h5")
VIDEO_MODEL_CANDIDATES = [
    VIDEO_MODEL_PATH,
    os.path.join(REPO_ROOT, "models", "video_model_best.keras"),
]

# The text model needs the SAME vocabulary it was trained with. After training,
# export it once with:
#     import pickle
#     pickle.dump(vectorizer.get_vocabulary(), open("text_vectorizer_vocab.pkl", "wb"))
# and drop that file here (or point to wherever you saved it).
TEXT_VOCAB_PATH = os.path.join(PROJECT_ROOT, "models", "text_vectorizer_vocab.pkl")
# ------------------------------------------------------------------

# Labels the models were trained on (must match training - edit if different).
EMOTION_LABELS = ["happy", "sad", "angry", "surprise", "disgust", "fear"]

# Text tokenizer settings (must match what was used during training).
TEXT_MAX_LEN = 50

# Audio feature settings (must match preprocessing/audio_preprocessing.py used at training time).
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 4.0
AUDIO_N_MFCC = 40
AUDIO_N_MELS = 64
AUDIO_HOP_LENGTH = 512
AUDIO_N_FFT = 1024
AUDIO_TIME_STEPS = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SEC / AUDIO_HOP_LENGTH) + 1

# Video feature settings (must match preprocessing/video_preprocessing.py used at training time).
VIDEO_NUM_FRAMES = 16
VIDEO_FRAME_SIZE = 96