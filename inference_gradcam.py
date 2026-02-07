import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import argparse
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers


def _iter_layers(module):
    for layer in getattr(module, "layers", []):
        yield layer
        if hasattr(layer, "layers") and layer.layers:
            yield from _iter_layers(layer)


def find_last_conv_layer(model):
    for layer in reversed(list(_iter_layers(model))):
        if isinstance(layer, layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in model.")


def preprocess_image_for_model(img_path, target_size, model_name_hint=None):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.resize(img, (int(target_size[1]), int(target_size[0])))
    batch = np.expand_dims(img.astype(np.float32), axis=0)
    # Choose preprocess based on model hint
    if model_name_hint and "efficientnet" in model_name_hint.lower():
        pre = getattr(tf.keras.applications, "efficientnet_v2", None)
        if pre:
            return pre.preprocess_input(batch)
    # fallback to MobileNetV2 preprocess if available
    mob = getattr(tf.keras.applications, "mobilenet_v2", None)
    if mob:
        return mob.preprocess_input(batch)
    # final fallback: scale to [-1,1]
    return batch / 127.5 - 1.0


def _build_grad_model(model, conv_layer_obj):
    # Rebuild a functional graph by re-calling the existing layers on a new Input.
    # This ensures the conv layer output tensor is connected to the model output.
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) < 4:
        raise ValueError("Unsupported model input shape.")

    inp = tf.keras.Input(shape=shape[1:], name="gradcam_input")

    # find parent layer that contains conv_layer_obj (for nested models)
    parent = None
    for layer in model.layers:
        if hasattr(layer, "layers") and layer.layers:
            try:
                nested = list(_iter_layers(layer))
            except Exception:
                nested = []
            if conv_layer_obj in nested:
                parent = layer
                break

    if parent is None:
        # conv layer lives in the top-level model
        conv_out = tf.keras.Model(inputs=model.inputs, outputs=conv_layer_obj.output)(inp)
        x = inp
        for layer in model.layers:
            x = layer(x)
        return tf.keras.Model(inputs=inp, outputs=[conv_out, x])

    # conv layer is inside a nested model (e.g., EfficientNetV2S)
    base = parent
    # build a single base model that outputs both the conv activation and base output
    base_cam = tf.keras.Model(
        inputs=base.input,
        outputs=[base.get_layer(conv_layer_obj.name).output, base.output]
    )
    conv_out, base_out = base_cam(inp)

    # apply the remaining layers after the base
    try:
        base_idx = model.layers.index(base)
    except ValueError:
        base_idx = 0
    x = base_out
    for layer in model.layers[base_idx + 1:]:
        x = layer(x)

    return tf.keras.Model(inputs=inp, outputs=[conv_out, x])


def compute_gradcam(model, img_array, conv_layer_obj, target_class_index=None):
    grad_model = _build_grad_model(model, conv_layer_obj)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            if target_class_index is None:
                target_class_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, target_class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None; cannot compute Grad-CAM.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap_on_image(img_path, heatmap, out_path, alpha=0.45):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, overlay)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model .h5")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Output path for gradcam overlay")
    args = p.parse_args()

    model_path = args.model
    image_path = args.image
    output_path = args.output

    if not os.path.isfile(model_path):
        # try to find any .h5 or .keras files in the repo root as suggestions
        candidates = []
        for root, _, files in os.walk("."):
            for f in files:
                if f.lower().endswith(('.h5', '.keras')):
                    candidates.append(os.path.join(root, f))
        msg = f"Model file not found: {model_path}\n"
        if candidates:
            msg += "Found these model files in workspace:\n"
            msg += "\n".join(candidates[:20])
        msg += "\nPass a valid model path with --model"
        raise FileNotFoundError(msg)

    model = tf.keras.models.load_model(model_path)

    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if shape and len(shape) >= 3 and shape[1] and shape[2]:
        target_size = (int(shape[1]), int(shape[2]))
    else:
        target_size = (224, 224)

    model_name_hint = getattr(model, "name", "")
    img_array = preprocess_image_for_model(image_path, target_size, model_name_hint=model_name_hint)
    preds = model.predict(img_array)

    # determine label and score
    if preds.ndim == 1:
        prob = float(preds[0])
        label = "real" if prob >= 0.5 else "fake"
        score = prob
    else:
        last = preds.shape[-1]
        if last == 1:
            prob = float(preds[0][0])
            label = "real" if prob >= 0.5 else "fake"
            score = prob
        elif last == 2:
            idx = int(np.argmax(preds[0]))
            label = "real" if idx == 1 else "fake"
            score = float(preds[0][idx])
        else:
            idx = int(np.argmax(preds[0]))
            label = f"class_{idx}"
            score = float(preds[0][idx])

    result = {"label": label, "score": float(score) if score is not None else None, "gradcam_path": None}

    if label == "fake":
        conv_layer = find_last_conv_layer(model)
        heatmap = compute_gradcam(model, img_array, conv_layer)
        gradcam_path = overlay_heatmap_on_image(image_path, heatmap, output_path)
        result["gradcam_path"] = gradcam_path
        result["conv_layer"] = conv_layer.name

    print(json.dumps(result))


if __name__ == "__main__":
    main()