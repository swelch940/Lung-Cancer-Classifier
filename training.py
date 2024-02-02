import tensorflow as tf
import numpy 
import os
import PIL
from PIL import Image
import cv2
import imghdr
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from skimage import io
from keras.applications.densenet import DenseNet121
from keras.applications.resnet import ResNet50
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import random
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout



data_dir = "The IQ-OTHNCCD lung cancer dataset-segmentated"
dataset = tf.keras.utils.image_dataset_from_directory(data_dir,image_size=(224,224))

train_size = int(len(dataset)*.7)
val_size = int(len(dataset)*.20)
test_size = int(len(dataset)*.10)

train = dataset.take(train_size)
val = dataset.take(val_size)
test = dataset.take(test_size)

input_shape= (224,224, 3)

base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()

model.add(base_model)

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

logdir = "logs"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

hist = model.fit(train, epochs = 20, validation_data = val, callbacks = [tensorboard_callback])

model.evaluate(test)