import os #Sirve para leer carpetas del sistema
import matplotlib.pyplot as plt # grafica img o métricas
from tensorflow.keras.preprocessing.image import ImageDataGenerator #generador de img que las lee en el disco, las redimensiona, las normaliza y las entrega por batch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional

DATASET_PATH = 'dataset/' #apunta al directorio con las img, están guardadas en carpetas que corresponden a clases

num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'categorical'
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, #generador de img con normalización
                                   rotation_range=20, #se da cuenta de que una imagen inclinada dentro del rango de rotación (-20° y +20°) sigue siendo la misma clase
                                   #se da de cuenta que la orientación NO define la clase
                                   width_shift_range=0.1, #Mueve la img horizontalmente hasta un 10% de anchura, una de 128px puede moverse +12px
                                   # y el modelo aprende que el objeto no necesariamente debe estar centrado
                                   #adempas, aprende invariancia a posición horizontal
                                   height_shift_range=0.1, #hace lo mismo pero verticalmente, evita que no reconozca una img duplicada porque está más arriba
                                   zoom_range=0.1, #tolera el tamaño
                                   # zoom in y ou en +10%, aprende que el tamaño no define la clase y mejora la generalización
                                   #así, cuando una img esté más cerca o lejos y sea duplicada, pueda entender que ya la conoce
                                   horizontal_flip=True #efecto espejo, anque el objeto de la img esté volteado sigue siendo la misma clase, pero solo sirve con imágenes, no con texto
                                   )
train_data = train_datagen.flow_from_directory(
    DATASET_PATH, #ruta de la carpeta con las img
    target_size=(96, 96), #tamaño de las img
    batch_size=32, #nro de img en la iteración
    class_mode=class_mode, #modo binary o categorical
    subset='training' #80% de datos para entrenarlo
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(96, 96),
    batch_size=32,
    class_mode=class_mode,
    subset='validation'  #20% de los datos para la validación
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)

# 👇 ESTA es la capa CLAVE para Grad-CAM
x = tf.keras.layers.Conv2D(
    256, (3, 3),
    activation="relu",
    name="last_conv"
)(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
#--------------------------------------------------------------------------------------
#base_model lleva dentro lo siguiente: 'Input(shape=(96, 96, 3)), #alto, ancho, canales, imágenes son RGB → 3 canales

    #Conv2D(32, (3, 3), activation='relu'), #num de filtros (32), aprende patrones
    #BatchNormalization(), #estabiliza entrenamiento
    #MaxPooling2D(2, 2), #reduce dimensionalidad'
#---------------------------------------
#   NOTAS:
    #Flatten reemplazado porque MobileNet produce mapas especiales, menos parámetros y por ende menor overfitting y mejor generalización
    #Dropout evita que por ejemplos específicos crea que ya se las sabe todas (reduce la dependencia de neuronas específicas y mejora la generalización)


#compilación del modelo
model.compile(
    optimizer=Adam(learning_rate=1e-4), #LR pequeña para que el CNN pueda aprender en pasos pequeños (0.001) pero generalizando mejor y la validación se estabilida
    loss='categorical_crossentropy',
    metrics=['accuracy'])

early_stop = EarlyStopping( #ejerce controlsobre overfitting, paro el entrenamiento antes de que se autosabotee
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

#entrenamiento
history = model.fit(
          train_data, #datos para el entrenamiento
          validation_data=val_data, #para la validación
          epochs=50,
          callbacks=[early_stop])

def plot_training_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()
plot_training_curves(history)


#evaluación
test_loss, test_accuracy = model.evaluate(val_data)
print(f'Precisión en los datos de validación del modelo: {test_accuracy:.2f}')
model.save('Rebecca.h5')

