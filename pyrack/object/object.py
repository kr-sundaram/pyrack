import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from collections import Counter
import imageai
from imageai.Detection import ObjectDetection

def img(image):
    """Here we read the image as numpy arrays
    with the help of OpenCV"""
    global image_path
    image_path = image
    img = cv2.imread(image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def display_original_image(img):
    """Here we display the original image
    with the help of matplotlib"""
    a = plt.imshow(img)
    a = plt.suptitle('Original Image')
    a = plt.axis('off')
    return a

def model_weights(weights):
    """Here, you need to set the path
    where the pre-trained model
    weights are stored"""
    global model_path
    model_path = weights
    return model_path

def detections(img):
    """With the help of the pre-trained
    model weights, we detect all objects
    available in the image using imageai's
    object detection class"""
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(image_path, os.path.join(os.getcwd(), 'detections.jpg'))
    return detections

def detected_objects(img, detections):
    """Here we make a list of 
    all the objects detected in the image"""
    detected_objects = []
    for i in range(len(detections)):
      detected_objects.append(detections[i]['name'])
    return detected_objects
    
def bbox(detections):
    """Here we extract the bounding
    boxes of each detected object"""
    #global bbox
    bbox = []
    for i in range(len(detections)):
      bbox.append(detections[i]['box_points'])
    return bbox

def number_of_detections(img, detections):
    if len(detections) == 1:
      return '1 object detected'
    else:
      return '{} objects detected'.format(len(detections))

def detected_objects_shape(img = img, detected_objects = detected_objects, roi = roi):
    if len(detected_objects) == 0:
      return '0 objects detected'
    else:
      roi_shape = []
      for i in roi:
        roi_shape.append(i.shape)
      df = pd.DataFrame(columns = ['Object', 'Shape'])
      df['Object'] = detected_objects
      df['Shape'] = roi_shape
      return df
	  
def display_all_objects(img = img, detected_objects = detected_objects, roi = roi):
    if len(detected_objects) == 0:
      return '0 objects detected'
    elif len(detected_objects) == 1:
      a = plt.imshow(roi[0])
      a = plt.suptitle('The only {} detected in the image'.format(detected_objects[0]))
      a = plt.axis('off')
      a = plt.show()
      return a
    else:
      fig, ax = plt.subplots(1, len(detected_objects))
      for i in range(len(detected_objects)):
        a = ax[i].imshow(roi[i])
        a = ax[i].set_title(detected_objects[i])
        a = ax[i].axis('off')
      a = fig.suptitle('All objects detected in the image')
      a = plt.show()
      return a