import cv2
from PIL import Image
import numpy as np

class Overlay:
    def __init__(self):
        self.gifs = {}
        self.current_frames = {}
        self.frame_counts = {}

    def load_gif(self, name, path, target_size=None):
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

            if target_size:
                frames = [cv2.resize(frame, target_size) for frame in frames]

            self.gifs[name] = frames
            self.current_frames[name] = 0
            self.frame_counts[name] = len(frames)
            print(f"Loaded GIF {name} with {len(frames)} frames")

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

        alpha = overlay_frame[:, :, 3] / 255.0
        rgb_bgr = cv2.cvtColor(overlay_frame[:, :, :3], cv2.COLOR_RGB2BGR)
        result = frame.copy()

        alpha = alpha[:, :, np.newaxis]
        result = ((1 - alpha) * result + alpha * rgb_bgr).astype(np.uint8)

        self.current_frames[name] = (current_frame_index + 1) % self.frame_counts[name]

        return result
    
    