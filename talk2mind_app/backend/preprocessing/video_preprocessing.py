"""
video_preprocessing.py (inference copy)
=========================================
Same frame-sampling + face-crop pipeline used at training time. Must stay in
sync with whatever produced video_model_final.h5's training data.
"""

import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_paths as mp

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _crop_largest_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return frame_bgr
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return frame_bgr[y:y + h, x:x + w]


def sample_frame_indices(total_frames, num_samples=mp.VIDEO_NUM_FRAMES):
    if total_frames <= 0:
        return np.zeros(num_samples, dtype=int)
    return np.round(np.linspace(0, total_frames - 1, num_samples)).astype(int)


def extract_face_sequence(video_path, num_frames=mp.VIDEO_NUM_FRAMES, frame_size=mp.VIDEO_FRAME_SIZE):
    """Returns np.ndarray of shape (num_frames, frame_size, frame_size, 3), float32 in [0,1]."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total, num_frames).tolist())

    grabbed = {}
    frame_i = 0
    while cap.isOpened() and len(grabbed) < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i in indices:
            face = _crop_largest_face(frame)
            face = cv2.resize(face, (frame_size, frame_size))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            grabbed[frame_i] = face.astype(np.float32) / 255.0
        frame_i += 1
    cap.release()

    ordered = [grabbed[i] for i in sorted(grabbed.keys())]
    if len(ordered) == 0:
        ordered = [np.zeros((frame_size, frame_size, 3), dtype=np.float32)]
    while len(ordered) < num_frames:
        ordered.append(ordered[-1])
    return np.stack(ordered[:num_frames], axis=0)