import streamlit as st
import cv2
import numpy as np
import joblib
import time
import io
import csv
import threading
from collections import Counter
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing_module
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import pyttsx3

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="ISL Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4aa;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .title-box h1 { color: #00d4aa; font-size: 2rem; margin: 0; }
    .title-box p  { color: #aaaaaa; margin: 5px 0 0 0; }

    .gesture-box {
        background: #1a1a2e;
        border: 2px solid #00d4aa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .gesture-label  { color: #aaaaaa; font-size: 0.9rem; margin-bottom: 5px; }
    .gesture-word   { color: #00d4aa; font-size: 2rem; font-weight: bold; }
    .confidence-high { color: #00ff88; font-size: 1rem; }
    .confidence-mid  { color: #ffcc00; font-size: 1rem; }
    .confidence-low  { color: #ff4444; font-size: 1rem; }

    .sentence-box {
        background: #1a1a2e;
        border: 2px solid #4a9eff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        min-height: 80px;
    }
    .sentence-label { color: #aaaaaa; font-size: 0.9rem; margin-bottom: 8px; }
    .sentence-text  { color: #ffffff; font-size: 1.4rem; font-weight: bold; word-wrap: break-word; }

    .stat-box {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stat-number { color: #00d4aa; font-size: 1.8rem; font-weight: bold; }
    .stat-label  { color: #aaaaaa; font-size: 0.8rem; }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        padding: 10px;
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

# ================= SESSION STATE =================
if "sentence"        not in st.session_state: st.session_state.sentence        = []
if "last_word"       not in st.session_state: st.session_state.last_word       = ""
if "word_count"      not in st.session_state: st.session_state.word_count      = 0
if "last_gesture"    not in st.session_state: st.session_state.last_gesture    = ""
if "last_confidence" not in st.session_state: st.session_state.last_confidence = 0.0

# ================= VIDEO PROCESSOR =================
class ISLProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp_hands_module
        self.hands    = mp_hands_module.Hands(
            max_num_hands=2,
            min_detection_confidence=0.3
        )
        self.mp_draw         = mp_drawing_module
        self.current_preds   = []
        self.last_added_time = 0
        self.delay           = 1.5
        self.lock            = threading.Lock()

    def recv(self, frame):
        img     = frame.to_ndarray(format="bgr24")
        img     = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result  = self.hands.process(rgb)

        predicted_word = ""
        confidence_val = 0.0
        data_aux       = []

        # ================= LANDMARK EXTRACTION =================
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks[:2]:
                x_ = [lm.x for lm in handLms.landmark]
                y_ = [lm.y for lm in handLms.landmark]
                for lm in handLms.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))
                self.mp_draw.draw_landmarks(
                    img, handLms,
                    mp_hands_module.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 212, 170), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

            # Pad to 84
            if len(data_aux) == 42:
                data_aux += [0.0] * 42

            if len(data_aux) == 84:
                try:
                    arr            = np.asarray(data_aux).reshape(1, -1)
                    predicted_word = model.predict(arr)[0]
                    proba          = model.predict_proba(arr)[0]
                    confidence_val = np.max(proba)
                except Exception as e:
                    print("Prediction error:", e)

            if predicted_word and confidence_val > 0.4:
                self.current_preds.append(predicted_word)

        # ================= STABILIZE =================
        if len(self.current_preds) > 10:
            final_pred   = Counter(self.current_preds).most_common(1)[0][0]
            current_time = time.time()

            if final_pred != st.session_state.last_word and \
               (current_time - self.last_added_time) > self.delay:
                with self.lock:
                    st.session_state.sentence.append(final_pred)
                    st.session_state.last_word       = final_pred
                    st.session_state.word_count      += 1
                    st.session_state.last_gesture    = final_pred
                    st.session_state.last_confidence = confidence_val
                self.last_added_time = current_time

            self.current_preds = []

        # ================= OVERLAY =================
        color = (0, 255, 136) if confidence_val > 0.8 else \
                (255, 204, 0)  if confidence_val > 0.6 else (255, 68, 68)

        cv2.rectangle(img, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(img,
                    f"{predicted_word}  ({confidence_val*100:.0f}%)",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================= UI =================
st.markdown("""
<div class="title-box">
    <h1>🤟 ISL Sign Language Translator</h1>
    <p>Indian Sign Language → Text → Voice | Real-Time AI Recognition</p>
</div>
""", unsafe_allow_html=True)

col_cam, col_ctrl = st.columns([3, 2])

# ================= CAMERA COLUMN =================
with col_cam:
    st.markdown("### 📷 Live Camera Feed")
    ctx = webrtc_streamer(
        key="isl-translator",
        video_processor_factory=ISLProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ================= CONTROLS COLUMN =================
with col_ctrl:
    st.markdown("### 🎯 Recognition Output")

    word       = st.session_state.last_gesture
    conf       = st.session_state.last_confidence
    conf_class = "confidence-high" if conf > 0.8 else \
                 "confidence-mid"  if conf > 0.6 else "confidence-low"

    st.markdown(f"""
    <div class="gesture-box">
        <div class="gesture-label">Current Gesture</div>
        <div class="gesture-word">{word if word else "—"}</div>
        <div class="{conf_class}">{conf*100:.0f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    full_sentence = " ".join(st.session_state.sentence)
    st.markdown(f"""
    <div class="sentence-box">
        <div class="sentence-label">📝 Sentence</div>
        <div class="sentence-text">{full_sentence if full_sentence else "Start signing..."}</div>
    </div>
    """, unsafe_allow_html=True)

    # ================= BUTTONS =================
    st.markdown("### 🎮 Controls")
    b1, b2 = st.columns(2)

    with b1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.sentence        = []
            st.session_state.last_word       = ""
            st.session_state.word_count      = 0
            st.session_state.last_gesture    = ""
            st.session_state.last_confidence = 0.0
            st.rerun()

    with b2:
        if st.button("⬅️ Backspace", use_container_width=True):
            if st.session_state.sentence:
                st.session_state.sentence.pop()
                st.session_state.word_count = max(0, st.session_state.word_count - 1)
                st.rerun()

    if st.button("🔊 Speak Sentence", use_container_width=True):
        if full_sentence.strip():
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(full_sentence)
            engine.runAndWait()
        else:
            st.warning("Nothing to speak yet!")

    # ================= DOWNLOAD =================
    st.markdown("### ⬇️ Download")

    txt_content = f"""ISL Sign Language Translator
==============================
Date    : {time.strftime("%Y-%m-%d %H:%M:%S")}
Team    : AIML-141 | SRMIST
==============================

Sentence:
{full_sentence if full_sentence else "No sentence recorded."}

Words Recognized : {len(st.session_state.sentence)}
Total Detected   : {st.session_state.word_count}
==============================
"""

    csv_buffer = io.StringIO()
    writer     = csv.writer(csv_buffer)
    writer.writerow(["S.No", "Word", "Timestamp"])
    for i, w_item in enumerate(st.session_state.sentence, 1):
        writer.writerow([i, w_item, time.strftime("%H:%M:%S")])
    csv_content = csv_buffer.getvalue()

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="📄 TXT",
            data=txt_content,
            file_name=f"isl_output_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not bool(full_sentence.strip())
        )

    with d2:
        st.download_button(
            label="📊 CSV",
            data=csv_content,
            file_name=f"isl_output_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not bool(full_sentence.strip())
        )

    # ================= STATS =================
    st.markdown("### 📊 Session Stats")
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(st.session_state.sentence)}</div>
            <div class="stat-label">Words</div>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{st.session_state.word_count}</div>
            <div class="stat-label">Total Recognized</div>
        </div>""", unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{conf*100:.0f}%</div>
            <div class="stat-label">Confidence</div>
        </div>""", unsafe_allow_html=True)

# ================= SUPPORTED GESTURES =================
st.markdown("---")
st.markdown("### 🤟 Supported Gestures")

gestures = [
    "BEAUTIFUL", "DRINK", "DRIVE", "FOOTBALL", "GO",
    "HOME", "I", "CHENNAI", "LIKE", "MILK",
    "OUTSIDE", "PLAY", "RAIN", "SEE", "SHE",
    "TODAY", "WENT", "YOU", "HOW_ARE_YOU", "WHAT_ARE_YOU_DOING"
]

cols = st.columns(5)
for i, gesture in enumerate(gestures):
    with cols[i % 5]:
        st.markdown(f"""
        <div style="background:#1a1a2e; border:1px solid #00d4aa;
                    border-radius:8px; padding:8px; text-align:center;
                    margin-bottom:8px; color:#00d4aa; font-size:0.85rem;">
            🤟 {gesture}
        </div>
        """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.85rem;">
    ISL Recognition System | Team AIML-141 | SRM Institute of Science and Technology<br>
    Mohamed Sarfraz & Rupha Sri M | Guide: Dr. Sreekrishna M
</div>
""", unsafe_allow_html=True)