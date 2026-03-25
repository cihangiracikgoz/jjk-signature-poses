import os
import numpy as np
from constants import SAMPLES_PATH, LANDMARK_VECTOR_SIZE

class GestureDetector:
    def __init__(self):
        self.samples = []
        self.labels = []

    def add_sample(self, vector, label):
        self.samples.append(vector)
        self.labels.append(label)

    def save_samples(self, path=SAMPLES_PATH):
        if len(self.samples) == 0:
            print("No samples to save.")
            return

        X_new = np.array(self.samples)
        y_new = np.array(self.labels)

        if os.path.exists(path):
            exist = np.load(path)
            X_new = np.concatenate([exist['X'], X_new])
            y_new = np.concatenate([exist['y'], y_new])

        np.savez(path, X=X_new, y=y_new)
        print(f"Saved {len(self.samples)} new samples to {path} (total: {len(y_new)})")

    def get_landmark_vector(self, hand_landmarks, handedness):
        if hand_landmarks is None or handedness is None:
            return np.zeros(LANDMARK_VECTOR_SIZE, dtype=np.float32)

        left = np.zeros((21, 3), dtype=np.float32)
        right = np.zeros((21, 3), dtype=np.float32)

        for landmark_set, hand_info in zip(hand_landmarks, handedness):
            label = hand_info.classification[0].label

            coordinates = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmark_set.landmark], 
                dtype=np.float32
            )

            coordinates -= coordinates[0]
            scale = np.linalg.norm(coordinates[9])  
            if scale > 0:
                coordinates /= scale

            if label == 'Left':
                left = coordinates
            else:                
                right = coordinates
        
        vector = np.concatenate([left.flatten(), right.flatten()])
        return vector