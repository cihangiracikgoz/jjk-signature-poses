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
                    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGR)
                    frames.append(frame_bgr)
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

    def apply_overlay(self, frame, name):
        if name not in self.gifs or self.gifs[name] == []:
            return frame
        
        gif_frames = self.gifs[name]
        current_frame_index = self.current_frames[name]
        overlay_frame = gif_frames[current_frame_index]

        gif_resized = cv2.resize(overlay_frame, (frame.shape[1], frame.shape[0]))
        result = cv2.addWeighted(frame, 0.5, gif_resized, 0.5, 0)
        self.current_frames[name] = (current_frame_index + 1) % self.frame_counts[name]

        return result
    
    