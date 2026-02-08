import numpy as np

class GestureDetector:
    def __init__(self):
        self.samples = []
        self.labels = []

    def add_sample(self, vector, label):
        self.samples.append(vector)
        self.labels.append(label)

    def save_samples(self, path="gesture_samples.npz"):
        np.savez(
            path,
            X=np.array(self.samples),
            y=np.array(self.labels)
        )
        print(f"Saved {len(self.samples)} samples to {path}")

    def get_landmark_vector(self, hand_landmarks, handedness):
        if hand_landmarks is None or handedness is None:
            return np.zeros(126, dtype=np.float32)

        left = np.zeros((21, 3), dtype=np.float32)
        right = np.zeros((21, 3), dtype=np.float32)

        for landmark_set, hand_info in zip(hand_landmarks, handedness):
            label = hand_info.classification[0].label

            coordinates = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmark_set.landmark], 
                dtype=np.float32
            )

            if label == 'Left':
                left = coordinates
            else:                
                right = coordinates
        
        vector = np.concatenate([left.flatten(), right.flatten()])
        return vector