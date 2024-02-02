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
from scipy import ndimage
from PIL import Image as im
import matplotlib.image

def retrive_images(dir, name):
   return os.listdir(os.path.join(dir,name))

def image_to_array(cases):
    images = []

    for image in cases:
        imgage = Image.open("The IQ-OTHNCCD lung cancer dataset/Bengin cases/" + image).convert(
        "L")
        imgs = numpy.array(image)
        images.append(imgs)
    return images


def median_filter(dataset):
    filtered_images = []

    for image in dataset:
        imgage = ndimage.median_filter(image, size = 3)
        filtered_images.append(image)
    
    return filtered_images


def new_folders(path, filtered_images, cases):
     
     while i < filtered_images:
        cv2.imwrite(os.path.join(path, cases + str(i) + ".jpg"), filtered_images[i])
        i+=1

def scale(data, name):
    scaled = []
    for i in data:
        if i == ".ipynb_checkpoints":
            continue
        image = cv2.imread("The IQ-OTHNCCD lung cancer dataset-processed/" + name + "/"+ i)
        image -= image.min() 
        image = image / image.max()
        image *= 255 # [0, 255] range
        cv2.imwrite("The IQ-OTHNCCD lung cancer dataset-scaled/" + name  +"/" + i, image)
    

def pre_process_dataset_for_training():
    bengin_cases = retrive_images("The IQ-OTHNCCD lung cancer dataset", "Bengin cases")
    normal_cases = retrive_images("The IQ-OTHNCCD lung cancer dataset", "Normal cases")
    malignant_cases = retrive_images("The IQ-OTHNCCD lung cancer dataset", "Malignant cases")

    bengin_dataset = image_to_array(bengin_cases)
    normal_dataset = image_to_array(normal_cases)
    malignant_dataset = image_to_array(malignant_cases)

    filtered_bengin = median_filter(bengin_dataset)
    filtered_normal = median_filter(normal_dataset)
    filtered_malignant = median_filter(malignant_dataset)

    new_folders("The IQ-OTHNCCD lung cancer dataset-processed", filtered_bengin, "Bengin cases")
    new_folders("The IQ-OTHNCCD lung cancer dataset-processed", filtered_normal, "Normal cases")
    new_folders("The IQ-OTHNCCD lung cancer dataset-processed", filtered_malignant, "Malignant cases")


    bengin_cases = retrive_images("The IQ-OTHNCCD lung cancer dataset-processed", "Bengin cases")
    normal_cases = retrive_images("The IQ-OTHNCCD lung cancer dataset-processed", "Normal cases")
    malignant_cases = retrive_images("The IQ-OTHNCCD lung cancer dataset-processed", "Malignant cases")


    scale(bengin_cases, "Bengin cases")
    scale(normal_cases, "Normal cases")
    scale(malignant_cases, "Malignant cases")


    scaled_bengin = retrive_images("The IQ-OTHNCCD lung cancer dataset-scaled", "Bengin cases")
    scaled_normal = retrive_images("The IQ-OTHNCCD lung cancer dataset-scaled", "Normal cases")
    scaled_malignant  = retrive_images("The IQ-OTHNCCD lung cancer dataset-scaled", "Malignant cases")


    return scaled_bengin, scaled_normal, scaled_malignant

def pre_procees_image():
    return



data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])


