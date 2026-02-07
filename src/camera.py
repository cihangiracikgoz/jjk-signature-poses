import mediapipe as mp
import cv2

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        return frame
    
    def hand_detection(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.hands.process(rgb_frame)

        if output.multi_hand_landmarks:
            for hand_landmarks in output.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        if output.multi_hand_landmarks:
            return output.multi_hand_landmarks
        else:
            return None
        
    def release(self):
        self.cap.release()
    
    
        
        
        
        