import cv2
import joblib
import numpy as np
from src.camera import Camera
from src.gesture_detector import GestureDetector
from src.overlay import Overlay
from collections import deque
from constants import LABEL_NAMES, GIF_MAPPING, MODEL_PATH, THRESHOLD, SMOOTHING

def main():
    model = joblib.load(MODEL_PATH)

    with Camera() as camera:
        gesture_detector = GestureDetector()
        gif_overlay = Overlay()

        for _, (name, path) in GIF_MAPPING.items():
            gif_overlay.load_gif(name, path)

        current_gif = None
        app_name = "Jujutsu Kaisen - Gesture Recognition"

        label = 0
        confidence = 0.0
        buffer = deque(maxlen=SMOOTHING)

        cap_width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow(app_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(app_name, cap_width, cap_height)

        while True:
            frame = camera.capture_frame()
            if frame is None:
                break

            hands, handedness = camera.hand_detection(frame)
            vector = gesture_detector.get_landmark_vector(hands, handedness)

            if hands is not None:
                probabilities = model.predict_proba([vector])[0]
                raw_label = int(np.argmax(probabilities))
                raw_confidence = probabilities[raw_label]
            else:
                raw_label = 0
                raw_confidence = 0.0

            buffer.append((raw_label, raw_confidence))

            if len(buffer) == SMOOTHING:
                label = int(np.bincount([item[0] for item in buffer]).argmax())
                confidence = np.mean([item[1] for item in buffer if item[0] == label])
            else:
                label = raw_label
                confidence = raw_confidence

            if confidence > THRESHOLD and label in GIF_MAPPING:
                new_gif = GIF_MAPPING[label][0]
                if new_gif != current_gif:
                    current_gif = new_gif
                    gif_overlay.reset_gif(current_gif)
            else:
                current_gif = None

            if current_gif:
                frame = gif_overlay.apply_overlay(frame, current_gif)

            cv2.putText(
                frame,
                f"{LABEL_NAMES[label]} ({confidence:.0%})",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(app_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


