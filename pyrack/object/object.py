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
  
def roi(img, detections):
    bbox = []
    roi = []
    for i in range(len(detections)):
      bbox.append(detections[i]['box_points'])
    for i in bbox:
      for (x,y,w,h) in [i]:
        cropped_img = img[y:y+h, x:x+w]
        for (h, w, d) in [cropped_img.shape]:
          if h > 100 and w > 100:
            roi.append(cropped_img)
          elif h > 100 and w < 100:
            cropped_img = cv2.copyMakeBorder(cropped_img, 0, 0, (150 - w), (150 - w), cv2.BORDER_CONSTANT, value = (0,0,0))
            cropped_img = cv2.putText(cropped_img, 'Too small to be displayed', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            roi.append(cropped_img)
          elif h < 100 and w > 100:
            cropped_img = cv2.copyMakeBorder(cropped_img, (150 - h), (150 - h), 0, 0, cv2.BORDER_CONSTANT, value = (0,0,0))
            cropped_img = cv2.putText(cropped_img, 'Too small to be displayed', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            roi.append(cropped_img)
          else:
            cropped_img = cv2.copyMakeBorder(cropped_img, (150 - h), (150 - h), (150 - w), (150 - w), cv2.BORDER_CONSTANT, value = (0,0,0))
            cropped_img = cv2.putText(cropped_img, 'Too small to be displayed', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            roi.append(cropped_img)
    return roi
	
def resized_roi(img = img, roi = roi):
    resized_roi = []
    for cropped_img in roi:
      resized_cropped_img = cv2.resize(cropped_img, (100,100))
      resized_cropped_img = cv2.rectangle(resized_cropped_img, (0,0), (100,100), (0,0,0), 3)
      resized_roi.append(resized_cropped_img)
    return resized_roi

def unique_items_detected(img, detected_objects):
    df = pd.DataFrame(columns = ['detected_objects'])
    df['detected_objects'] = detected_objects
    unique_items_detected = df['detected_objects'].unique()
    if len(unique_items_detected) == 0:
      return '0 objects detected'
    elif len(unique_items_detected) == 1:
      return '{} unique object detected - {}'.format(len(unique_items_detected), (unique_items_detected))
    else:
      return '{} unique objects detected - {}'.format(len(unique_items_detected), (unique_items_detected))

def count_per_unique_item(img = img, detected_objects = detected_objects):
    if len(detected_objects) == 0:
      return '0 objects detected'
    elif len(detected_objects) == 1:
      return 'Only 1 instance of {} detected'.format(detected_objects[0])
    else:
      df = pd.DataFrame(columns = ['Unique Item'])
      df['Unique Item'] = detected_objects
      df['Count'] = df['Unique Item'].map(df['Unique Item'].value_counts(dropna = False))
      df = df.drop_duplicates(subset = ['Unique Item', 'Count'], keep = 'first')
      df = df.reset_index(drop = True)
      a = df.plot.barh(x = 'Unique Item', y = 'Count')
      a = plt.suptitle('Count of each unique object detected', size = 15)
      a = plt.show()
      return a
    
def number_of_detections(img, detections):
    if len(detections) == 1:
      return '1 object detected'
    else:
      return '{} objects detected'.format(len(detections))
