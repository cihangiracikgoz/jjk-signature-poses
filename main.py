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

        hand_landmarks = camera.hand_detection(frame)

if __name__ == "__main__":
    main()


