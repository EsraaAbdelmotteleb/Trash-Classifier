# -*- coding: utf-8 -*-
"""
wasteClassifier.ipynb

This file is adapted from TensorFlowLite’s tutorial named Recognize Flowers with TensorFlow on Android. Modifications have been made.
A copy of TensorFlowLite's License can be found at
https://www.apache.org/licenses/LICENSE-2.0



from __future__ import absolute_import, division, print_function, unicode_literals

!pip install tf-nightly-gpu-2.0-preview
import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

tf.__version__


[link text](https://)Download the flowers dataset.
"""

## Setup 

_URL = "https://github.com/EsraaAbdelmotteleb/AndroTrash/releases/download/v1/dataset-resized.zip"

zip_file = tf.keras.utils.get_file(origin=_URL, 
                                   fname="dataset-resized.zip", 
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'dataset-resized')

"""
Rescale the images using ImageDataGenerator 
Create the training generator 
Specify image size, batch size and the directory of training dataset directory.
Create the validation generator

"""

IMAGE_SIZE = 224
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')

for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape

"""
Save the labels into a file
"""
print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(labels)

!cat labels.txt

""""
Create the base model from the pre-trained model

"""

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

"""
Extract features
a.  Un-freeze the top layers of the model
b.  Train the weights of the newly unfrozen layers alongside training the weights of the top-level classifier
c.  Compile the model 
d.  Train the top-level classifier.
"""

base_model.trainable = False


model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(32, 3, kernel_regularizer=tf.keras.regularizers.l2(0.35),activation='relu'),
  tf.keras.layers.Dropout(0.27),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(6, kernel_regularizer=tf.keras.regularizers.l2(0.1),activation='softmax'),
])




model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))



epochs = 10

history = model.fit(train_generator, 
                    epochs=epochs)

"""
Learning curves

"""

acc = history.history['accuracy']

loss = history.history['loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()

"""
Fine tune the model
a.  Un-freeze the top layers of the model
b.  Train the weights of the newly unfrozen layers alongside training the weights of the top-level classifier

"""

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False



model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(1e-5),
              metrics=['accuracy'])

model.summary()

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))



history_fine = model.fit(train_generator, 
                         epochs=5)

testing = model.evaluate(val_generator)

"""
Convert to TFLite

"""

saved_model_dir = 'save/fine_tuning'
tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

"""Download the converted model and labels"""

from google.colab import files

files.download('model.tflite')
files.download('labels.txt')


""" Results """

"""Learning curves of the training  accuracy/loss"""

acc = history_fine.history['accuracy']

loss = history_fine.history['loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training  Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()
