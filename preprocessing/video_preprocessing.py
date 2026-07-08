"""
video_preprocessing.py
========================
Samples config.VIDEO_NUM_FRAMES evenly-spaced frames from a video clip,
detects/crops the face in each (falls back to full frame if no face found),
resizes to config.VIDEO_FRAME_SIZE, and normalizes to [0, 1].

Uses OpenCV's built-in Haar Cascade for face detection (no extra
dependency). Swap in MTCNN/RetinaFace for production-grade accuracy.
"""

import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _crop_largest_face(frame_bgr):
    """Detect and crop the largest face in a frame; fall back to full frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return frame_bgr
    # pick the largest detected face box
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return frame_bgr[y:y + h, x:x + w]


def sample_frame_indices(total_frames, num_samples=config.VIDEO_NUM_FRAMES):
    """Evenly spaced frame indices across the clip (handles short clips gracefully)."""
    if total_frames <= 0:
        return np.zeros(num_samples, dtype=int)
    if total_frames <= num_samples:
        # repeat frames if the clip is shorter than the required sequence length
        idx = np.linspace(0, total_frames - 1, num_samples)
    else:
        idx = np.linspace(0, total_frames - 1, num_samples)
    return np.round(idx).astype(int)


def extract_face_sequence(video_path, num_frames=config.VIDEO_NUM_FRAMES,
                           frame_size=config.VIDEO_FRAME_SIZE):
    """
    Extract a fixed-length sequence of normalized, face-cropped frames.

    Returns
    -------
    np.ndarray of shape (num_frames, frame_size, frame_size, 3), dtype float32, range [0,1]
    """
    cap = cv2.VideoCapture(video_path)
    try:
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
                # Keep as uint8 to save 4x memory during extraction
                grabbed[frame_i] = face
            frame_i += 1
    finally:
        cap.release()

    # Preserve temporal order; pad with the last valid frame if some indices were missed
    ordered = [grabbed[i] for i in sorted(grabbed.keys())]
    if len(ordered) == 0:
        ordered = [np.zeros((frame_size, frame_size, 3), dtype=np.uint8)]
    while len(ordered) < num_frames:
        ordered.append(ordered[-1])
    ordered = ordered[:num_frames]

    # Convert to float32 and normalize at the very end to minimize float32 memory footprint
    return np.stack(ordered, axis=0).astype(np.float32) / 255.0

