"""
main.py
=======
FastAPI backend for Talk2Mind's 3-model detection app.

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /predict/text   { "text": "..." }                -> emotion scores
    POST /predict/audio  multipart file upload (.wav)      -> emotion scores
    POST /predict/video  multipart file upload (.mp4/.mov) -> emotion scores
    GET  /health                                           -> readiness check
"""

import os
import sys
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import model_paths as mp
from inference.text_infer import predict_text
from inference.audio_infer import predict_audio
from inference.video_infer import predict_video

app = FastAPI(title="Talk2Mind Detection API")

# Allow the Streamlit frontend (running on a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    """Quick check that all 3 model files exist at the configured paths."""
    return {
        "text_model_found": any(os.path.exists(path) for path in mp.TEXT_MODEL_CANDIDATES),
        "audio_model_found": os.path.exists(mp.AUDIO_MODEL_PATH),
        "video_model_found": os.path.exists(mp.VIDEO_MODEL_PATH),
        "text_vocab_found": os.path.exists(mp.TEXT_VOCAB_PATH),
    }


@app.post("/predict/text")
def predict_text_endpoint(payload: TextRequest):
    scores = predict_text(payload.text)
    return {"modality": "text", "scores": scores}


@app.post("/predict/audio")
def predict_audio_endpoint(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        scores = predict_audio(tmp_path)
    finally:
        os.remove(tmp_path)
    return {"modality": "audio", "scores": scores}


@app.post("/predict/video")
def predict_video_endpoint(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        scores = predict_video(tmp_path)
    finally:
        os.remove(tmp_path)
    return {"modality": "video", "scores": scores}