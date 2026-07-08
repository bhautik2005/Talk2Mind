"""
dataset_loader.py
==================
Reads the CMU-MOSEI-style label CSVs (Data_Train_modified.csv etc.) and
builds synchronized (text, audio, video) samples for a given split, matching
the folder layout you showed:

  Audio_chunk/Train_modified/<file>.wav
  Audio_chunk/Labels/Data_Train_modified.csv
  Video_Dataset/Train/<file>.mp4

Each CSV row (one utterance/clip) is joined to its audio file and video file
by video_id + clip_id, so the same row drives all three modalities -
this is what keeps the fusion training synchronized across modalities.
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.audio_preprocessing import extract_features as extract_audio_features
from preprocessing.video_preprocessing import extract_face_sequence


def load_label_dataframe(split):
    """Load and lightly validate the label CSV for a split ('train'/'val'/'test')."""
    csv_path = config.SPLIT_CSV[split]
    df = pd.read_csv(csv_path)

    required = [config.COL_VIDEO_ID, config.COL_START_TIME, config.COL_END_TIME, config.COL_TEXT] + config.EMOTION_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing expected columns {missing}. "
            f"Update config.py column-name constants to match your actual headers: "
            f"found columns = {list(df.columns)}"
        )
    return df


def _build_paths(df, split):
    """Attach resolved audio_path / video_path columns to the dataframe, dropping
    rows whose media file can't be found on disk (keeps train/eval robust to gaps)."""
    audio_dir = config.SPLIT_AUDIO_DIR[split]
    video_dir = config.SPLIT_VIDEO_DIR[split]

    video_dir_exists = os.path.isdir(video_dir)
    audio_paths, video_paths, keep_mask = [], [], []
    for _, row in df.iterrows():
        vid = row[config.COL_VIDEO_ID]
        start = row[config.COL_START_TIME]
        end = row[config.COL_END_TIME]
        a_path = os.path.join(audio_dir, config.AUDIO_FILENAME_TEMPLATE.format(video_id=vid, start_time=start, end_time=end))
        v_path = os.path.join(video_dir, config.VIDEO_FILENAME_TEMPLATE.format(video_id=vid, start_time=start, end_time=end))
        ok = os.path.exists(a_path) and (not video_dir_exists or os.path.exists(v_path))
        audio_paths.append(a_path)
        video_paths.append(v_path)
        keep_mask.append(ok)

    df = df.copy()
    df["audio_path"] = audio_paths
    df["video_path"] = video_paths
    kept = df[np.array(keep_mask)].reset_index(drop=True)
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[dataset_loader] Warning: dropped {dropped}/{len(df)} rows in split "
              f"'{split}' with missing audio/video files.")
    return kept


class Talk2MindDataset:
    """
    Loads label CSV + media paths for a split, and exposes:
      - a pandas dataframe (self.df) for inspection
      - a tf.data.Dataset yielding ((text_tokens, audio_feat, video_frames), label)
        for use directly with model.fit()

    Text tokenization uses a pre-fit `text_vectorizer` (a
    tf.keras.layers.TextVectorization instance) so vocabulary is consistent
    across train/val/test - fit it once on the TRAIN split's text and reuse it.
    """

    def __init__(self, split, text_vectorizer=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.df = _build_paths(load_label_dataframe(split), split)
        self.text_vectorizer = text_vectorizer

    def set_text_vectorizer(self, vectorizer):
        self.text_vectorizer = vectorizer

    # ---- per-sample loading (runs in a tf.py_function so librosa/opencv can be used) ----
    def _load_sample(self, idx, modalities):
        idx = int(idx.numpy())
        row = self.df.iloc[idx]

        labels = row[config.EMOTION_COLUMNS].values.astype(np.float32)

        text = ""
        if 0 in modalities:
            text = str(row[config.COL_TEXT])

        if 1 in modalities:
            audio_feat = extract_audio_features(row["audio_path"])
        else:
            # dummy audio features
            audio_feat = np.zeros((config.AUDIO_TIME_STEPS, config.AUDIO_N_MFCC + config.AUDIO_N_MELS + 2), dtype=np.float32)

        if 2 in modalities:
            video_path = row["video_path"]
            cache_path = video_path.replace(".mp4", "_face_seq.npy")
            if os.path.exists(cache_path):
                try:
                    video_frames = np.load(cache_path)
                except Exception:
                    video_frames = extract_face_sequence(video_path)
                    try:
                        np.save(cache_path, video_frames)
                    except Exception:
                        pass
            else:
                video_frames = extract_face_sequence(video_path)
                try:
                    np.save(cache_path, video_frames)
                except Exception as e:
                    print(f"Warning: Could not save video cache for {video_path}: {e}")
        else:
            # dummy video frames
            video_frames = np.zeros((config.VIDEO_NUM_FRAMES, config.VIDEO_FRAME_SIZE, config.VIDEO_FRAME_SIZE, 3), dtype=np.float32)

        res_text = text
        res_audio = audio_feat.astype(np.float32)
        res_video = video_frames.astype(np.float32)
        res_labels = labels
        
        # Clean up temporary references
        del text, audio_feat, video_frames, labels
        gc.collect()

        return res_text, res_audio, res_video, res_labels

    def _tf_load_sample(self, idx, modalities):
        text, audio_feat, video_frames, labels = tf.py_function(
            func=lambda i: self._load_sample(i, modalities), inp=[idx],
            Tout=[tf.string, tf.float32, tf.float32, tf.float32],
        )
        text.set_shape([])
        audio_feat.set_shape([config.AUDIO_TIME_STEPS, None])
        video_frames.set_shape([config.VIDEO_NUM_FRAMES, config.VIDEO_FRAME_SIZE,
                                 config.VIDEO_FRAME_SIZE, 3])
        labels.set_shape([len(config.EMOTION_COLUMNS)])

        if self.text_vectorizer is not None and 0 in modalities:
            tokens = self.text_vectorizer(tf.expand_dims(text, 0))[0]
        else:
            tokens = tf.zeros([config.TEXT_MAX_LEN], dtype=tf.int64)

        return (tokens, audio_feat, video_frames), labels

    def as_tf_dataset(self, batch_size=config.BATCH_SIZE, shuffle=True, modalities=(0, 1, 2)):
        n = len(self.df)
        ds = tf.data.Dataset.range(n)
        if shuffle:
            ds = ds.shuffle(buffer_size=n, reshuffle_each_iteration=True)
        
        # Limit parallel calls when loading video frames to 1 to prevent OOM
        num_parallel = 1 if 2 in modalities else tf.data.AUTOTUNE
        
        ds = ds.map(lambda idx: self._tf_load_sample(idx, modalities), num_parallel_calls=num_parallel)
        ds = ds.batch(batch_size).prefetch(2)
        return ds

    def texts(self):
        """All raw transcript strings in this split (used to fit the TextVectorization layer)."""
        return self.df[config.COL_TEXT].astype(str).tolist()

    def __len__(self):
        return len(self.df)
