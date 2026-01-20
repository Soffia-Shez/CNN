import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np

MODEL_PATH = 'Rebecca.h5'
DATASET_PATH = 'dataset/' #apunta al directorio con las img, están guardadas en carpetas que corresponden a clases
IMG_SIZE = (96, 96)
LAST_CONV_LAYER = 'Conv_1'

class_names = sorted(os.listdir(DATASET_PATH))

model = tf.keras.models.load_model(MODEL_PATH)
print('Modelo cargado correctamente.')

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
        model.get_layer("last_conv").output,  # mapas de activación
        model.output                          # predicciones
    ]
)


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def make_gradcam_heatmap(img_array, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    # 🔥 NORMALIZACIÓN AGRESIVA
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.power(heatmap, 0.3)

    return heatmap

def overlay_gradcam(img_path, heatmap, alpha=0.6):
    # Imagen original
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    # Heatmap
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = heatmap.astype("float32") / 255.0

    # Superposición REAL
    superimposed = heatmap * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)

    return superimposed

def predict_with_gradcam(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)

    class_index = np.argmax(preds[0])
    confidence = preds[0][class_index]
    class_name = class_names[class_index]

    heatmap = make_gradcam_heatmap(img_array, class_index)
    cam_image = overlay_gradcam(img_path, heatmap, alpha=0.45)

    plt.figure(figsize=(6, 6))
    plt.imshow(cam_image)
    plt.title(
        f"Predicción: {class_name}\n"
        f"Certeza: {confidence:.2%}"
    )
    plt.axis("off")
    plt.show()

    print("Probabilidades:")
    for c, p in zip(class_names, preds[0]):
        print(f"{c}: {p:.4f}")

predict_with_gradcam("dataset/Delfines/delfin9.jpg")
predict_with_gradcam("dataset/Tortugas/tortuga7.jpg")
predict_with_gradcam("dataset/Pulpos/pulpo5.jpg")


