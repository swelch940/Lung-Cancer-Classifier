
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import CTForm
from .models import CTSCAN
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from numpy import asarray
from scipy import ndimage
import PIL
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model
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
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import random
from sklearn.model_selection import train_test_split

import keras

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras.models import load_model
import json


# Create your views here.
@ensure_csrf_cookie
def process_variable(request):
    if request.method == 'POST':
        # Retrieve the JavaScript variable from the AJAX request
        json_data = json.loads(request.body.decode('utf-8'))
        variable_value = json_data.get('variable_name')
       
   
        
        variable_value = numpy.array(variable_value)
       
        #image = image_to_array(variable_value)
        image_float32 = variable_value.astype('float32')
        image_rgb = cv2.cvtColor(image_float32, cv2.COLOR_BGR2RGB)
        
        image = median_filter(image_rgb)

        #image = normalize(image)
        print(image)
        image = segmentation(image)
        image= tf.image.resize(image, (224,224))
        print(image)
        result = load_model(Xception, "webpage/X_UNSEG.h5", np.expand_dims(image/255, 0))
        print(result)

        # Process the variable (perform any necessary computation)
        #print(processed_value)
        # Return the processed value as a JSON response
        return JsonResponse({'processed_value': result})

def image_to_array(image):
   
    image = Image.open(image).convert(
        "L")
    imgs = np.array(image)
    
    return imgs


def median_filter(image):
    return ndimage.median_filter(image, size = 3)

def normalize(image):
    return image / 255

def segmentation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    #the below line of code defines the criteria for the algorithm to stop running, 
    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy) 
    #becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
    # then perform k-means clustering with number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
 
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    return segmented_image


def load_model(name, weights,image ):

    data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])
    input_shape = (224,224,3)
    base_model = name(weights=None, include_top=False, input_shape=input_shape)
    name = Sequential()
    name.add(data_augmentation)
    name.add(base_model)
    name.add(Flatten())
    name.add(Dense(224, activation='relu'))
    name.add(Dense(1, activation='sigmoid'))

    name.compile(
    optimizer="adam",
       loss=tf.keras.losses.binary_crossentropy,
    metrics=["accuracy"],
)
    name.load_weights(weights, by_name = True, skip_mismatch = True)
    

    if name.predict(image) > 0.5:
        return "Cancer!"
    else:
        return "Not Cancer!"

    

def home(request):
    if request.method == 'POST':
        form = CTForm(request.POST, request.FILES)
        #form = UploadFileForm(request.POST, request.FILES)
        
        if form.is_valid():
            form.save()
            image = image_to_array(form)
            
        return redirect('success')
    else:
        form = CTForm()
    return render(request, 'base.html', {'form': form})

def ct_image_view(request):
 
    if request.method == 'POST':
        form = CTForm(request.POST, request.FILES)
 
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = CTForm()
    return render(request, 'base.html', {'form': form})
 

def pipeline(request):
    return

def success(request):

    return HttpResponse('successfully uploaded')

def result(request):
    return HttpResponse('Result')