"""
streamlit_app.py
=================
Talk2Mind — 3-section (Text / Voice / Face) mental well-being check-in.

Run with:
    streamlit run streamlit_app.py

Talks to the FastAPI backend (default http://localhost:8000) — start that
first with:
    uvicorn main:app --reload --port 8000
"""

import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# >>>>>>>>>>>>>>>>  EDIT THIS IF YOUR BACKEND RUNS ELSEWHERE  <<<<<<<<<<<<<<<<
BACKEND_URL = "http://localhost:8000"
# ------------------------------------------------------------------

EMOTION_LABELS = ["happy", "sad", "angry", "surprise", "disgust", "fear"]
# Which labels count toward "distress" when we compute the summary score.
NEGATIVE_LABELS = ["sad", "angry", "disgust", "fear"]

st.set_page_config(page_title="Talk2Mind Check-in", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
if "text_prediction" not in st.session_state:
    st.session_state["text_prediction"] = None
if "voice_prediction" not in st.session_state:
    st.session_state["voice_prediction"] = None
if "face_prediction" not in st.session_state:
    st.session_state["face_prediction"] = None

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def distress_score(scores: dict) -> float:
    """Simple 0-100 well-being score: 100 = great, 0 = high distress.
    Customize this formula to match how your models were actually trained/labeled."""
    if not scores:
        return 50.0
    neg = np.mean([scores.get(l, 0.0) for l in NEGATIVE_LABELS])
    pos = scores.get("happy", 0.0)
    raw = 0.5 + (pos - neg) / 2  # roughly maps to [0,1]
    return float(np.clip(raw, 0, 1) * 100)


def gauge(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4C6EF5"},
            "steps": [
                {"range": [0, 33], "color": "#FFD6D6"},
                {"range": [33, 66], "color": "#FFF3BF"},
                {"range": [66, 100], "color": "#D3F9D8"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ------------------------------------------------------------------
# Page Presenters
# ------------------------------------------------------------------
def show_dashboard():
    st.header("📊 Your Mental Well-Being Dashboard")

    text_scores = st.session_state["text_prediction"]
    voice_scores = st.session_state["voice_prediction"]
    face_scores = st.session_state["face_prediction"]

    s_text = distress_score(text_scores) if text_scores else None
    s_voice = distress_score(voice_scores) if voice_scores else None
    s_face = distress_score(face_scores) if face_scores else None

    # Calculate overall score
    valid_scores = [s for s in [s_text, s_voice, s_face] if s is not None]
    overall = float(np.mean(valid_scores)) if valid_scores else None

    st.subheader("Overall Mental Wellness")
    if overall is not None:
        st.plotly_chart(gauge(overall, "Overall Wellness"), use_container_width=True)
    else:
        st.info("No prediction available (please complete at least one check-in first)")

    st.subheader("Modality Scores")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Text Emotion")
        if s_text is not None:
            st.plotly_chart(gauge(s_text, "Text"), use_container_width=True)
        else:
            st.info("No prediction available")

    with c2:
        st.markdown("### Voice Emotion")
        if s_voice is not None:
            st.plotly_chart(gauge(s_voice, "Voice"), use_container_width=True)
        else:
            st.info("No prediction available")

    with c3:
        st.markdown("### Face Emotion")
        if s_face is not None:
            st.plotly_chart(gauge(s_face, "Face"), use_container_width=True)
        else:
            st.info("No prediction available")

    if valid_scores:
        st.subheader("Emotion Breakdown by Modality")
        rows = []
        if text_scores:
            for label in EMOTION_LABELS:
                rows.append({"Modality": "Text", "Emotion": label, "Score": text_scores.get(label, 0.0)})
        if voice_scores:
            for label in EMOTION_LABELS:
                rows.append({"Modality": "Voice", "Emotion": label, "Score": voice_scores.get(label, 0.0)})
        if face_scores:
            for label in EMOTION_LABELS:
                rows.append({"Modality": "Face", "Emotion": label, "Score": face_scores.get(label, 0.0)})
        
        if rows:
            df = pd.DataFrame(rows)
            st.bar_chart(df.pivot(index="Emotion", columns="Modality", values="Score"))

        st.subheader("Recommendations")
        if overall >= 66:
            st.success("You're showing largely positive indicators today. Keep up whatever's working — "
                        "consistent sleep, movement, and connection with people you trust.")
        elif overall >= 33:
            st.warning("Some signs of stress or low mood came through. Consider a short walk, a break from "
                        "screens, or talking to someone you trust today.")
        else:
            st.error("Your responses show signs of significant distress. Please consider reaching out to a "
                      "mental health professional or a trusted person soon. If you're in crisis, contact a "
                      "local crisis helpline immediately.")

        with st.expander("Raw scores (debug)"):
            st.json({
                "text": text_scores,
                "voice": voice_scores,
                "face": face_scores
            })


def show_text_page():
    st.header("🗣️ Step 1 — Text Check-in")
    st.caption("Takes about 1–2 minutes")

    prompt = st.text_area(
        "How have the last few days been for you? What's on your mind right now?",
        height=150,
        placeholder="Write a few honest sentences — there's no right answer...",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Submit & Analyze", type="primary", disabled=len(prompt.strip()) < 5)

    if submit:
        with st.spinner("Analyzing your response..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/predict/text", json={"text": prompt}, timeout=30)
                resp.raise_for_status()
                st.session_state["text_prediction"] = resp.json()["scores"]
                st.success("Text analysis complete! You can view the results on the Dashboard.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}. Is `uvicorn main:app` running? ({e})")


def show_voice_page():
    st.header("🎙️ Step 2 — Voice Check-in")
    st.caption("Takes about 1–2 minutes. Speak naturally for 10–20 seconds.")

    st.write("Prompt: *\"Tell me about how your energy and sleep have been lately.\"*")

    if hasattr(st, "audio_input"):
        audio_value = st.audio_input("Record your response")
    elif hasattr(st, "experimental_audio_input"):
        audio_value = st.experimental_audio_input("Record your response")
    else:
        st.warning("Audio input is not supported in your Streamlit version. Please upgrade Streamlit or use the file uploader below.")
        audio_value = None
    uploaded_audio = st.file_uploader("...or upload a .wav file instead", type=["wav"])

    audio_bytes, filename = None, "recording.wav"
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
    elif uploaded_audio is not None:
        audio_bytes = uploaded_audio.getvalue()
        filename = uploaded_audio.name

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Submit & Analyze", type="primary", disabled=audio_bytes is None)

    if submit:
        with st.spinner("Analyzing your voice..."):
            try:
                files = {"file": (filename, audio_bytes, "audio/wav")}
                resp = requests.post(f"{BACKEND_URL}/predict/audio", files=files, timeout=60)
                resp.raise_for_status()
                st.session_state["voice_prediction"] = resp.json()["scores"]
                st.success("Voice analysis complete! You can view the results on the Dashboard.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}. ({e})")


def show_face_page():
    st.header("🎥 Step 3 — Face Check-in")
    st.caption("Takes about 1–2 minutes. A short clip of your face is enough.")

    st.write("Prompt: *\"Describe one thing that made you smile this week — and one thing that stressed you out.\"*")

    uploaded_video = st.file_uploader("Upload a short video clip (.mp4 / .mov)", type=["mp4", "mov", "avi"])
    st.caption("Tip: record a 10–20 second selfie clip on your phone and upload it here.")

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Submit & Analyze", type="primary", disabled=uploaded_video is None)

    if submit:
        with st.spinner("Analyzing your facial expressions..."):
            try:
                files = {"file": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")}
                resp = requests.post(f"{BACKEND_URL}/predict/video", files=files, timeout=120)
                resp.raise_for_status()
                st.session_state["face_prediction"] = resp.json()["scores"]
                st.success("Face analysis complete! You can view the results on the Dashboard.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}. ({e})")


 


# ------------------------------------------------------------------
# Navigation Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Talk2Mind")
    st.caption("A ~5–6 minute multimodal well-being check-in")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "💬 Text Check-in",
            "🎤 Voice Check-in",
            "😊 Face Check-in",
        ]
            
    )

    st.markdown("---")
    if st.button("🔄 Clear All Predictions", use_container_width=True):
        st.session_state["text_prediction"] = None
        st.session_state["voice_prediction"] = None
        st.session_state["face_prediction"] = None
        st.success("Predictions cleared!")
        st.rerun()

# ------------------------------------------------------------------
# Main Routing Logic
# ------------------------------------------------------------------
if page == "🏠 Dashboard":
    show_dashboard()
elif page == "💬 Text Check-in":
    show_text_page()
elif page == "🎤 Voice Check-in":
    show_voice_page()
elif page == "😊 Face Check-in":
    show_face_page()
 