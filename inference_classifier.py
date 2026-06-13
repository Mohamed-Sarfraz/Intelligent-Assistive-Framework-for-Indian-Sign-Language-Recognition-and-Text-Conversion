import joblib
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
from collections import Counter

# ================= LOAD MODEL =================
model = joblib.load("model.pkl")
print(f"Model loaded | Expected features: {model.n_features_in_}")

# ================= TEXT TO SPEECH =================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
mp_draw  = mp.solutions.drawing_utils

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# ================= VARIABLES =================
sentence            = []
current_predictions = []
last_added_time     = 0
delay               = 1.5
last_word           = ""

# ================= HELPERS =================
def wrap_text(text, max_chars=45):
    if len(text) <= max_chars:
        return text, ""
    split = text.rfind(' ', 0, max_chars)
    if split == -1:
        split = max_chars
    return text[:split], text[split:]

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = hands.process(rgb)

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
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Always pad to 84 to match retrained model
        if len(data_aux) == 42:
            data_aux += [0.0] * 42

        if len(data_aux) == 84:
            try:
                arr            = np.asarray(data_aux).reshape(1, -1)
                predicted_word = model.predict(arr)[0]
                proba          = model.predict_proba(arr)[0]
                confidence_val = np.max(proba)
                print(f"Predicted: {predicted_word} | Confidence: {confidence_val*100:.1f}%")
            except Exception as e:
                print("Prediction error:", e)

        if predicted_word and confidence_val > 0.4:
            current_predictions.append(predicted_word)

    # ================= STABILIZE & ADD TO SENTENCE =================
    if len(current_predictions) > 10:
        final_pred   = Counter(current_predictions).most_common(1)[0][0]
        current_time = time.time()

        if final_pred != last_word and (current_time - last_added_time) > delay:
            sentence.append(final_pred)
            last_word       = final_pred
            last_added_time = current_time

        current_predictions = []

    # ================= CONTROLS =================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        sentence            = []
        last_word           = ""
        current_predictions = []

    elif key == ord('b'):
        if sentence:
            sentence.pop()

    elif key == ord('s'):
        full = " ".join(sentence)
        if full.strip():
            engine.say(full)
            engine.runAndWait()

    elif key == ord('q'):
        break

    # ================= DISPLAY =================
    cv2.rectangle(frame, (0, 0), (w, 130), (20, 20, 20), -1)

    color = (0, 255, 0) if confidence_val > 0.8 else (0, 200, 255) if confidence_val > 0.6 else (0, 0, 255)
    cv2.putText(frame, f"Gesture: {predicted_word}  ({confidence_val*100:.0f}%)",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    full_sentence = " ".join(sentence)
    line1, line2  = wrap_text(full_sentence)
    cv2.putText(frame, f"Sentence: {line1}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)
    if line2:
        cv2.putText(frame, line2, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)

    cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "C: Clear  |  B: Backspace  |  S: Speak  |  Q: Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("ISL Sign Language Translator", frame)

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
