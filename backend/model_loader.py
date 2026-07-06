from tensorflow.keras.models import load_model
from config import MODEL_PATH


def get_model():
    """
    Load and return the trained CNN model.
    """
    model = load_model(MODEL_PATH)
    return model

if __name__ == "__main__":
    model = get_model()
    print("Model loaded successfully")
    print(model.input_shape)
    print(model.output_shape)