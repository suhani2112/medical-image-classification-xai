"""Prediction utilities placeholder."""
from model_loader import get_model
from config import CLASS_NAMES


model = get_model()


def predict_image(input_image):
    prediction = model.predict(input_image, verbose=0)

    pneumonia_probability = float(prediction[0][0])

    if pneumonia_probability > 0.5:
        label = CLASS_NAMES[1]  # Pneumonia
        confidence = pneumonia_probability * 100
    else:
        label = CLASS_NAMES[0]  # Normal
        confidence = (1 - pneumonia_probability) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "raw_probability": round(pneumonia_probability, 4)
    }
if __name__ == "__main__":
    from preprocess import load_image, preprocess_image

    image = load_image(
        r"D:\chest xray\chest_xray\test\PNEUMONIA\person100_bacteria_475.jpeg"
    )

    input_image = preprocess_image(image)

    result = predict_image(input_image)

    print(result)