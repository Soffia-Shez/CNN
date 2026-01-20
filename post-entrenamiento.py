#paso 1, imports
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing import image

DATASET_PATH = 'dataset/'
#paso 2, cargar el modelo
model = load_model('Rebecca.h5')
model.summary()
#paso 3, especificar la capa sobre la que se va a trabajar
last_conv_layer_name = 'last_conv'
#paso 4, mapas de activación y predicción final
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.output
    ]
)
#paso 5, qué zonas influyeron en su desición
def make_gradcam_heatmap(img_array, grad_model, class_index=None):

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

#paso 6, superponer
def save_and_display_gradcam(
    img_path,
    heatmap,
    output_path,
    alpha=0.4,
    show=True
):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))

    # Normalizar heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay correcto
    superimposed_img = cv2.addWeighted(
        img, 1 - alpha,
        heatmap_color, alpha,
        0
    )

    # Guardar
    cv2.imwrite(output_path, superimposed_img)

    # Mostrar (solo debug)
    if show:
        cv2.imshow("Grad-CAM", superimposed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#paso 7, ¡procesa el dataset automáticamente!
OUTPUT_DIR = 'gradcam_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(96, 96))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        heatmap = make_gradcam_heatmap(
            img_array,
            grad_model
        )

        output_path = os.path.join(output_class_dir, img_name)
        save_and_display_gradcam(img_path, heatmap, output_path)

