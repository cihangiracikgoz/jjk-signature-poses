import math
import time
import cv2
from src.camera import Camera
from src.gesture_detector import GestureDetector
from src.config import load_config

COUNTDOWN = 3.0
RECORDING = 3.0


def main():
    config = load_config()
    label_names = config["label_names"]
    label_keys = {ord(str(i)): i for i in label_names}

    with Camera() as camera:
        gesture_detector = GestureDetector()

        labels_help = ", ".join(f"{i}={name}" for i, name in label_names.items())
        max_key = max(label_names.keys())
        print(f"Press keys 0-{max_key} to start recording gestures. Press 'q' to quit without saving.")
        print(f"Labels: {labels_help}")
        print("Press 's' to save and quit.")

        current_label = None
        mode = 'IDLE'
        start_time = None

        while True: 
            frame  = camera.capture_frame()
            if frame is None:
                break

            hands, handedness = camera.hand_detection(frame)
            vector = gesture_detector.get_landmark_vector(hands, handedness)
            now = time.time()

            if mode == 'COUNTDOWN':
                remaining = COUNTDOWN - (now - start_time)
                if remaining <= 0:
                    mode = 'RECORDING'
                    start_time = now
                    print(f"Recording '{label_names[current_label]}'")
                else:
                    cv2.putText(
                        frame,
                        f"Get ready: {math.ceil(remaining)}", 
                        (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 255), 
                        2
                    )

            elif mode == 'RECORDING':
                elapsed = now - start_time
                if elapsed <= RECORDING:
                    if hands is not None:
                        gesture_detector.add_sample(vector, current_label)
                    cv2.putText(
                        frame,
                        f"Recording {label_names[current_label]}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2   
                    )
                else:
                    print(f"Finished recording '{label_names[current_label]}'. Total samples: {len(gesture_detector.samples)}")
                    mode = 'IDLE'
                    current_label = None

            if mode == 'IDLE':
                cv2.putText(
                    frame,
                    "Press 0-4 to record gesture",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Gesture Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting without saving.")
                break
            elif key == ord('s'):
                gesture_detector.save_samples()
                print("Samples saved.")
                break
            elif key in label_keys and mode == 'IDLE':
                current_label = label_keys[key]
                mode = 'COUNTDOWN'
                start_time = time.time()
                print(f"Starting countdown for '{label_names[current_label]}'")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()