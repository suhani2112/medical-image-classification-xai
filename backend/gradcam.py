"""Grad-CAM explainability utilities placeholder."""
import os
import cv2
import numpy as np
import tensorflow as tf

from model_loader import get_model
from config import IMAGE_SIZE


model = get_model()
LAST_CONV_LAYER_NAME = "conv2d_2"


def build_gradcam_models():
    last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)

    conv_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input
    start = False

    for layer in model.layers:
        if layer.name == LAST_CONV_LAYER_NAME:
            start = True
            continue

        if start:
            x = layer(x)

    classifier_model = tf.keras.models.Model(classifier_input, x)

    return conv_model, classifier_model


conv_model, classifier_model = build_gradcam_models()


def generate_gradcam(image, input_image, save_path):
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(input_tensor)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        loss = predictions[:, 0]

    gradients = tape.gradient(loss, conv_outputs)

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    original_resized = cv2.resize(original_image, IMAGE_SIZE)

    superimposed_image = cv2.addWeighted(
        original_resized,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(
        save_path,
        cv2.cvtColor(superimposed_image, cv2.COLOR_RGB2BGR)
    )

    return save_path

if __name__ == "__main__":
    from preprocess import load_image, preprocess_image

    image_path = r"D:\chest xray\chest_xray\test\PNEUMONIA\person100_bacteria_475.jpeg"

    image = load_image(image_path)
    input_image = preprocess_image(image)

    output_path = "../results/backend_gradcam_test.jpeg"

    saved_path = generate_gradcam(image, input_image, output_path)

    print("Saved Grad-CAM at:", saved_path)
