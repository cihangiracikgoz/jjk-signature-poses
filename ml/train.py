import pickle
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

LABEL_NAMES = {
    0: "IDLE",
    1: "GOJO",
    2: "SUKUNA",
    3: "CHOSO",
    4: "YUJI",
}

def train_model():
    data = np.load('gesture_samples.npz')
    X = data['X']
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values())))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()