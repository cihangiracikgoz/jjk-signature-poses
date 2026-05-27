import joblib
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from constants import LABEL_NAMES, MODEL_PATH, SAMPLES_PATH

def train_model():
    data = np.load(SAMPLES_PATH)
    X = data['X']
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values())))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()