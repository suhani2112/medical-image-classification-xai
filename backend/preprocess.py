"""Image preprocessing utilities placeholder."""
import cv2
import numpy as np

from config import IMAGE_SIZE


def load_image(image_path):
    """
    Load an image from disk.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return image


def preprocess_image(image):
    """
    Preprocess image for CNN prediction.
    """

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    resized_image = cv2.resize(
        gray_image,
        IMAGE_SIZE
    )

    normalized_image = resized_image / 255.0

    input_image = normalized_image.reshape(
        1,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        1
    )

    return input_image

if __name__ == "__main__":

    image = load_image(
        r"D:\chest xray\chest_xray\test\PNEUMONIA\person100_bacteria_475.jpeg"
    )

    processed = preprocess_image(image)

    print(image.shape)

    print(processed.shape)
