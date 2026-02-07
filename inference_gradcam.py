import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2

classes = ['REAL', 'FAKE']

def get_last_conv_layer_name(base_model):
    for layer in reversed(base_model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in base model.")

def make_gradcam_heatmap(img_array, model, conv_layer_name):
    base_model = model.layers[0]
    conv_layer = base_model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def apply_gradcam(image_path, model, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    conv_layer_name = get_last_conv_layer_name(model.layers[0])
    heatmap = make_gradcam_heatmap(img_array, model, conv_layer_name)
    image_bgr = cv2.imread(image_path)
    image_bgr = cv2.resize(image_bgr, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0)
    cv2.imwrite(output_path, overlay)
    return output_path, conv_layer_name

def predict_and_gradcam(image_path, model, output_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)
    pred_index = int(np.argmax(preds[0]))
    label = classes[pred_index]
    score = float(preds[0][pred_index])
    if label == "FAKE":
        gradcam_path, conv_layer = apply_gradcam(image_path, model, output_path=output_path)
        return {"label": label, "score": score, "gradcam_path": gradcam_path, "conv_layer": conv_layer}
    return {"label": label, "score": score, "gradcam_path": None, "conv_layer": None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model.h5")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="gradcam/output.jpg")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    result = predict_and_gradcam(args.image, model, output_path=args.output)
    print(result)

if __name__ == "__main__":
    main()
