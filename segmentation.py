import tensorflow as tf
import numpy as np
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

from Pre_processing import scaled_bengin, scaled_normal, scaled_malignant
from Pre_processing import new_folders

def segmentation(image, name):
    
    image = cv2.imread("The IQ-OTHNCCD lung cancer dataset-scaled/" + name + "/"+ image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    #the below line of code defines the criteria for the algorithm to stop running, 
    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy) 
    #becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
    # then perform k-means clustering with number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
 
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    return segmented_image



def segment_data(dataset, cases):
    segmentated = []

    for image in dataset:
        if image == ".ipynb_checkpoints":
            continue
        segmentated.append(segmentation(image, cases))

    return segmentated


segmented_bengin = segment_data(scaled_bengin, "Bengin cases")
segmented_normal = segment_data(scaled_normal, "Normal cases")
segmented_malignant = segment_data(scaled_malignant, "Malignant cases")

