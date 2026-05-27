LABEL_NAMES = {
    0: 'IDLE',
    1: 'GOJO',
    2: 'SUKUNA',
    3: 'CHOSO',
    4: 'YUJI',
}

GIF_MAPPING = {
    1: ('gojo', 'assets/gifs/gojo-domain-expansion.gif'),
    2: ('sukuna', 'assets/gifs/sukuna-domain-expansion.gif'),
    3: ('choso', 'assets/gifs/choso-cursed-technique.gif'),
    4: ('yuji', 'assets/gifs/yuji-black-flash.gif'),
}

THRESHOLD = 0.8
SMOOTHING = 10
CAM_INDEX = 0
LANDMARK_VECTOR_SIZE = 21 * 3 * 2  

MODEL_PATH = 'model.joblib'
SAMPLES_PATH = 'gesture_samples.npz'
