# Talk2Mind — 3-Model Detection App (FastAPI + Streamlit)

A ~5–6 minute, 3-section mental well-being check-in (Text → Voice → Face)
backed by your trained `text_model_final.h5`, `audio_model_final.h5`, and
`video_model_final.h5`.

## Folder structure

```
talk2mind_app/
├── models/                              <- PUT YOUR 3 .h5 FILES HERE
│   ├── text_model_final.h5
│   ├── audio_model_final.h5
│   ├── video_model_final.h5
│   └── text_vectorizer_vocab.pkl        <- see "Text model note" below
│
├── backend/                             <- FastAPI
│   ├── model_paths.py                   <- ★ EDIT MODEL PATHS HERE ★
│   ├── main.py                          <- FastAPI app / endpoints
│   ├── requirements.txt
│   ├── inference/
│   │   ├── text_infer.py
│   │   ├── audio_infer.py
│   │   └── video_infer.py
│   └── preprocessing/
│       ├── audio_preprocessing.py
│       └── video_preprocessing.py
│
├── frontend/                            <- Streamlit
│   ├── streamlit_app.py                 <- 3-section dashboard UI
│   └── requirements.txt
│
├── run_backend.bat / run_backend.sh
└── run_frontend.bat / run_frontend.sh
```

## 1. Where to put your model files (and where to change the path)

Drop your 3 files into `models/`:
```
talk2mind_app/models/text_model_final.h5
talk2mind_app/models/audio_model_final.h5
talk2mind_app/models/video_model_final.h5
```

If you'd rather keep them somewhere else, open **`backend/model_paths.py`**
— it's the *only* file you need to edit — and change these 3 lines:
```python
TEXT_MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "text_model_final.h5")
AUDIO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "audio_model_final.h5")
VIDEO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "video_model_final.h5")
```
to absolute paths, e.g. `"D:/models/text_model_final.h5"`.

### Text model note — you also need the vocabulary file
Your text model was trained on tokenized integer sequences produced by a
`TextVectorization` layer. The `.h5` file does **not** include that
vocabulary, so you must export it once, right after training:
```python
import pickle
pickle.dump(vectorizer.get_vocabulary(), open("text_vectorizer_vocab.pkl", "wb"))
```
and place `text_vectorizer_vocab.pkl` in `models/` (or wherever
`TEXT_VOCAB_PATH` in `model_paths.py` points). Without this file, text
predictions will fail with a clear error telling you what's missing.

### If your labels aren't `[happy, sad, angry, surprise, disgust, fear]`
Update `EMOTION_LABELS` in `backend/model_paths.py` to match your model's
actual output order.

## 2. Install dependencies

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Frontend (separate terminal / venv is fine too)
cd ../frontend
pip install -r requirements.txt
```

## 3. Run it — 2 terminals

**Terminal 1 — backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/health` — it should report all 3 model files
and the vocab file as `found: true`. Fix `model_paths.py` if not.

**Terminal 2 — frontend:**
```bash
cd frontend
streamlit run streamlit_app.py
```
This opens the dashboard at `http://localhost:8501`.

(Or just double-click `run_backend.bat` then `run_frontend.bat` on Windows.)

## 4. What the UI does

1. **Step 1 — Text Check-in** (~1-2 min): user types a free-text response to
   a prompt → sent to `/predict/text`.
2. **Step 2 — Voice Check-in** (~1-2 min): user records or uploads a short
   `.wav` → sent to `/predict/audio`.
3. **Step 3 — Face Check-in** (~1-2 min): user uploads a short video clip →
   sent to `/predict/video`.
4. **Step 4 — Dashboard**: combines all 3 modalities into gauge charts, a
   per-emotion breakdown bar chart, and a plain-language recommendation.

Total: ~5-6 minutes end to end, matching your original 3-section brief.

## 5. Customizing the well-being score formula
The combination logic (how emotion scores turn into a 0-100 "well-being
score") lives in `distress_score()` inside `frontend/streamlit_app.py`. It's
currently a simple placeholder (happy vs. sad/angry/disgust/fear average) —
replace it with whatever formula matches how your models were actually
trained/labeled, or with a proper 4th fusion model if you train one later
(the model_paths.py / inference/ pattern extends the same way).

## 6. Known limitations / next steps
- Face check-in currently accepts an **uploaded video file**, not a live
  webcam feed — add `streamlit-webrtc` if you want in-browser recording.
- Everything runs on CPU by default; see the earlier project notes on GPU
  setup if inference feels slow on `video_model_final.h5`.
- No database/auth here — this is single-session, in-memory only. Add a
  database layer (e.g. SQLite/Postgres) if you want to persist check-ins
  across sessions/users.