import math
import time, cv2
from src.camera import Camera
from src.gesture_detector import GestureDetector
from constants import LABEL_NAMES

COUNTDOWN = 3.0
RECORDING = 3.0

LABEL_KEYS = {
    ord('0'): 0,
    ord('1'): 1,
    ord('2'): 2,
    ord('3'): 3,
    ord('4'): 4,
}  

def main():
    camera = Camera()
    gesture_detector = GestureDetector()

    print("Press keys 0-4 to start recording gestures. Press 'q' to quit without saving.")
    print("Labels: 0=IDLE, 1=GOJO, 2=SUKUNA, 3=CHOSO, 4=YUJI")
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
                print(f"Recording '{LABEL_NAMES[current_label]}'")
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
                    f"Recording {LABEL_NAMES[current_label]}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2   
                )
            else:
                print(f"Finished recording '{LABEL_NAMES[current_label]}'. Total samples: {len(gesture_detector.samples)}")
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
        elif key in LABEL_KEYS and mode == 'IDLE':
            current_label = LABEL_KEYS[key]
            mode = 'COUNTDOWN'
            start_time = time.time()
            print(f"Starting countdown for '{LABEL_NAMES[current_label]}'")

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()