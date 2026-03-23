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

        gif_resized = cv2.resize(overlay_frame, (frame.shape[1], frame.shape[0]))

        alpha = gif_resized[:, :, 3] / 255.0
        rgb_bgr = cv2.cvtColor(gif_resized[:, :, :3], cv2.COLOR_RGB2BGR)
        result = frame.copy()

        for c in range(3):
            result[:, :, c] = (alpha * rgb_bgr[:, :, c] + (1 - alpha) * result[:, :, c])

        self.current_frames[name] = (current_frame_index + 1) % self.frame_counts[name]

        return result
    
    