import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import cv2

# Load an image and preprocess it
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Resize for MobileNetV2
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Function to get the class label
def get_class_label(preds):
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]  # Get top prediction
    class_label = decoded_preds[0][1]  # Extract class name
    return class_label

# Function to generate Grad-CAM heatmap
def compute_gradcam(model, img_array, class_index, conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]  # Loss for target class

    grads = tape.gradient(loss, conv_outputs)  # Compute gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    # Multiply feature maps by importance weights
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)  # Compute heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU activation
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap

# Overlay heatmap on image
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)  # Convert to 0-255 scale
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map

    # heatmap = np.expand_dims(heatmap, axis=-1)
    # heatmap = np.concatenate([heatmap, heatmap, heatmap], axis=-1)

    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)  # Blend images
    return superimposed_img

if __name__ == "__main__":
    # Load a pretrained model (MobileNetV2)
    model = tf.keras.models.load("/Users/tarunc/Personal/deep_busters/image_model.h5")
    # model.summary()

    # Example Usage
    img_path = "images/dog-2.jpg"  # Replace with your image path
    img_array = preprocess_image(img_path)

    # Get model predictions
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])  # Get class index
    class_label = get_class_label(preds)  # Get class label

    print(f"Predicted Class: {class_label} (Index: {class_index})")

    # Compute Grad-CAM heatmap
    heatmap = compute_gradcam(model, img_array, class_index)

    # Overlay heatmap on the original image
    output_img = overlay_heatmap(img_path, heatmap)

    # Save the heatmap
    cv2.imwrite("heatmap/2.jpg", output_img)