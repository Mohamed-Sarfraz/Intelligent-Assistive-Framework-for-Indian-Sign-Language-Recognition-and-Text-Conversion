import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ================= LOAD DATA =================
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data   = np.asarray(data_dict['data'])    # shape: (N, 84)
labels = np.asarray(data_dict['labels'])

print(f"Dataset : {data.shape[0]} samples | {data.shape[1]} features | {len(set(labels))} classes")
print(f"Classes : {sorted(set(labels))}")

# ================= SPLIT =================
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# ================= TRAIN =================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(x_train, y_train)

# ================= EVALUATE =================
y_pred = model.predict(x_test)
score  = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {score * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================= SAVE =================
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
print(f"   Expected input features: {model.n_features_in_}")