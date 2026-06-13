import os
import cv2
import time

DATA_DIR = './images'
os.makedirs(DATA_DIR, exist_ok=True)

labels = [
    "BEAUTIFUL", "DRINK", "DRIVE", "FOOTBALL", "GO",
    "HOME", "I", "CHENNAI", "LIKE", "MILK",
    "OUTSIDE", "PLAY", "RAIN", "SEE", "SHE",
    "TODAY", "WENT", "YOU",
    "HOW_ARE_YOU", "WHAT_ARE_YOU_DOING"
]
dataset_size      = 150
countdown_seconds = 3

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for label in labels:
    class_dir = os.path.join(DATA_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    print(f'\nCollecting data for: {label}')

    # Wait for Q to start
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Show "{label}" | Press Q to start', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ISL Data Collector', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Countdown
    start = time.time()
    while time.time() - start < countdown_seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        frame     = cv2.flip(frame, 1)
        remaining = countdown_seconds - int(time.time() - start)
        cv2.putText(frame, f'Starting in {remaining}...', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow('ISL Data Collector', frame)
        cv2.waitKey(25)

    # Capture
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'{label} | {counter + 1}/{dataset_size}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('ISL Data Collector', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

    print(f'✅ Done: {dataset_size} images saved for "{label}"')

cap.release()
cv2.destroyAllWindows()
print("\n✅ All classes collected successfully!")