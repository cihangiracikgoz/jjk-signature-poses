import cv2
from PIL import Image
import numpy as np

class Overlay:
    def __init__(self):
        self.gifs = {}
        self.current_frames = {}
        self.frame_counts = {}

    def load_gif(self, name, path):
        try:
            gif = Image.open(path)
            frames = []
            try:
                while True:
                    frame = gif.convert('RGBA')
                    frames.append(np.array(frame))
                    gif.seek(len(frames))
            except EOFError:
                pass

            self.gifs[name] = frames
            self.current_frames[name] = 0
            self.frame_counts[name] = len(frames)

        except FileNotFoundError:
            print(f"GIF file '{path}' not found.")
            self.gifs[name] = []
        except Exception as e:
            print(f"Error loading GIF '{name}': {e}")
            self.gifs[name] = []

    def reset_gif(self, name):
        if name in self.current_frames:
            self.current_frames[name] = 0

    def apply_overlay(self, frame, name):
        if not self.gifs.get(name):
            return frame

        gif_frames = self.gifs[name]
        current_frame_index = self.current_frames[name]
        overlay_frame = gif_frames[current_frame_index]

        h, w = frame.shape[:2]
        overlay_frame = cv2.resize(overlay_frame, (w, h))

        alpha = overlay_frame[:, :, 3:4] / 255.0
        rgb_bgr = cv2.cvtColor(overlay_frame[:, :, :3], cv2.COLOR_RGB2BGR)
        result = ((1 - alpha) * frame + alpha * rgb_bgr).astype(np.uint8)

        self.current_frames[name] = (current_frame_index + 1) % self.frame_counts[name]

        return result
    
    