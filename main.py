import cv2
import pickle
import numpy as np
from src.camera import Camera
from src.gesture_detector import GestureDetector
from src.overlay import Overlay

LABEL_NAMES = {
    0: 'IDLE',
    1: 'GOJO',
    2: 'SUKUNA',
    3: 'CHOSO',
    4: 'YUJI',
}

GIF_MAPPING = {
    1: 'gojo',
    2: 'sukuna',
    3: 'choso',
    4: 'yuji',
}

THRESHOLD = 0.8

def main():
    camera = Camera()
    gesture_detector = GestureDetector()
    gif_overlay = Overlay()

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    gif_overlay.load_gif('choso', 'assets/gifs/choso-cursed-technique.gif')
    gif_overlay.load_gif('gojo', 'assets/gifs/gojo-domain-expansion.gif')
    gif_overlay.load_gif('sukuna', 'assets/gifs/sukuna-domain-expansion.gif')
    gif_overlay.load_gif('yuji', 'assets/gifs/yuji-black-flash.gif')
    current_gif = None
    app_name = "Jujutsu Kaisen - Gesture Recognition"

    label = 0
    confidence = 0.0

    cv2.namedWindow(app_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(app_name, 1920, 1080)

    while True:
        frame = camera.capture_frame()
        if frame is None:
            break

        hands, handedness = camera.hand_detection(frame)
        vector = gesture_detector.get_landmark_vector(hands, handedness)

        if hands is not None:
            probabilities = model.predict_proba([vector])[0]
            label = int(np.argmax(probabilities))
            confidence = probabilities[label]
        else:
            label = 0
            confidence = 0.0

        if confidence > THRESHOLD and label in GIF_MAPPING:
            new_gif = GIF_MAPPING[label]
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

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


