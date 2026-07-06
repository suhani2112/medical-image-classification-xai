import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "chest_xray_cnn_model.keras"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Normal",
    "Pneumonia"
]

print(MODEL_PATH)
print(IMAGE_SIZE)
print(CLASS_NAMES)