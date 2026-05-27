import argparse
import os
import sys
import cv2
import joblib
import numpy as np
from src.camera import Camera
from src.gesture_detector import GestureDetector
from src.overlay import Overlay
from src.logger import logger, setup
from collections import deque
from src.config import load_config
from constants import MODEL_PATH, THRESHOLD, SMOOTHING, CAM_INDEX



def parse_args():
    parser = argparse.ArgumentParser(
        prog="gesture-recognition",
        description="Real-time hand gesture recognition with animated overlays",
    )
    parser.add_argument(
        "-c", "--camera", type=int, default=CAM_INDEX,
        help="camera device index (default: %(default)s)"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=THRESHOLD,
        help="confidence threshold for gesture activation (default: %(default)s)"
    )
    parser.add_argument(
        "-m", "--model", type=str, default=MODEL_PATH,
        help="path to trained model file (default: %(default)s)"
    )
    parser.add_argument(
        "-s", "--smoothing", type=int, default=SMOOTHING,
        help="number of frames for prediction smoothing (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="enable verbose logging output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    setup(args.verbose)

    config = load_config()
    label_names = config["label_names"]
    gif_mapping = config["gif_mapping"]

    if not os.path.exists(args.model):
        logger.error("Model not found at '%s'. Train one first: python ml/train.py", args.model)
        sys.exit(1)

    model = joblib.load(args.model)
    logger.info("Model loaded from %s", args.model)

    with Camera(args.camera) as camera:
        gesture_detector = GestureDetector()
        gif_overlay = Overlay()

        for _, (name, path) in gif_mapping.items():
            gif_overlay.load_gif(name, path)

        current_gif = None
        app_name = config["app_name"]

        label = 0
        confidence = 0.0
        buffer = deque(maxlen=args.smoothing)

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

            if len(buffer) == args.smoothing:
                label = int(np.bincount([item[0] for item in buffer]).argmax())
                confidence = np.mean([item[1] for item in buffer if item[0] == label])
            else:
                label = raw_label
                confidence = raw_confidence

            if confidence > args.threshold and label in gif_mapping:
                new_gif = gif_mapping[label][0]
                if new_gif != current_gif:
                    current_gif = new_gif
                    gif_overlay.reset_gif(current_gif)
            else:
                current_gif = None

            if current_gif:
                frame = gif_overlay.apply_overlay(frame, current_gif)

            cv2.putText(
                frame,
                f"{label_names[label]} ({confidence:.0%})",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(app_name, frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


