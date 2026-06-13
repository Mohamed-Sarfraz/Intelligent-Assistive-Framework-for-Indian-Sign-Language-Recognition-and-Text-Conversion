import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './images'

TWO_HAND_LABELS = {"HOW_ARE_YOU", "WHAT_ARE_YOU_DOING", "FOOTBALL"}

LABELS = [
    "BEAUTIFUL", "DRINK", "DRIVE", "FOOTBALL", "GO",
    "HOME", "I", "CHENNAI", "LIKE", "MILK",
    "OUTSIDE", "PLAY", "RAIN", "SEE", "SHE",
    "TODAY", "WENT", "YOU",
    "HOW_ARE_YOU", "WHAT_ARE_YOU_DOING"
]

data    = []
labels  = []
skipped = 0

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue

    if dir_ not in LABELS:
        print(f"Skipping unknown folder: {dir_}")
        continue

    is_two_hand      = dir_ in TWO_HAND_LABELS
    expected_features = 84 if is_two_hand else 42
    class_count      = 0

    print(f"Processing: {dir_} ({'2-hand' if is_two_hand else '1-hand'})")

    for img_path in sorted(os.listdir(dir_path)):
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            skipped += 1
            continue

        data_aux = []

        for handLms in results.multi_hand_landmarks[:2]:
            x_ = [lm.x for lm in handLms.landmark]
            y_ = [lm.y for lm in handLms.landmark]
            for lm in handLms.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # Pad to 84 for two-hand labels if only one hand detected
        if is_two_hand and len(data_aux) == 42:
            data_aux += [0.0] * 42

        if len(data_aux) == expected_features:
            data.append(data_aux)
            labels.append(dir_)
            class_count += 1
        else:
            skipped += 1

    print(f"  ✅ {class_count} samples collected for {dir_}")

# Pad ALL single-hand samples to 84 for consistent model input
data_padded = []
for sample, label in zip(data, labels):
    if len(sample) == 42:
        sample = sample + [0.0] * 42
    data_padded.append(sample)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data_padded, 'labels': labels}, f)

print(f"\n✅ Dataset saved!")
print(f"   Total samples : {len(data_padded)}")
print(f"   Total classes : {len(set(labels))}")
print(f"   Skipped       : {skipped}")