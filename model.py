import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers, Model, Sequential, optimizers, losses, metrics # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

classes = ['REAL', 'FAKE']

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)
validation_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb', shuffle=True
)

efficientnet = keras.applications.EfficientNetV2S(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False
)

efficientnet.trainable = True

Model = tf.keras.Sequential([
    efficientnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation='softmax')
])

Model.compile(optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=["accuracy"])

callbacks = [
    ModelCheckpoint("best_model.h5", verbose=1, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", verbose=1, factor=0.1, patience=25, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True, verbose=1)
]

model = Model.fit(train_generator, epochs=300, validation_data=validation_generator, callbacks=callbacks) 

plt.plot(model.history['accuracy'], label='train accuracy')
plt.plot(model.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()
