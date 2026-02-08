import cv2
from src.camera import Camera
from src.gesture_detector import GestureDetector
from src.overlay import Overlay

def main():
    camera = Camera()
    gesture_detector = GestureDetector()
    gif_overlay = Overlay()

    gif_overlay.load_gif('choso', 'assets/gifs/choso-cursed-technique.gif')
    gif_overlay.load_gif('gojo', 'assets/gifs/gojo-domain-expansion.gif')
    gif_overlay.load_gif('sukuna', 'assets/gifs/sukuna-domain-expansion.gif')
    gif_overlay.load_gif('yuji', 'assets/gifs/yuji-black-flash.gif')
    current_gif = None

    while True:
        frame = camera.capture_frame()
        if frame is None:
            break

        hands, handedness = camera.hand_detection(frame)
        vector = gesture_detector.get_landmark_vector(hands, handedness)
        #print(vector.shape)
        #assert vector.shape == (126,)

        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


