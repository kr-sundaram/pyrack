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
    """Here we extract the rigions of interest 
    across each detected object based on the
    bounding boxes returned above"""
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
            cropped_img = cv2.putText(cropped_img, 'Too small to be displayed accurately', (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cropped_img = cv2.putText(cropped_img, 'Kindly refer to the original image', (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            roi.append(cropped_img)
          elif h < 100 and w > 100:
            cropped_img = cv2.copyMakeBorder(cropped_img, (150 - h), (150 - h), 0, 0, cv2.BORDER_CONSTANT, value = (0,0,0))
            cropped_img = cv2.putText(cropped_img, 'Too small to be displayed accurately', (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cropped_img = cv2.putText(cropped_img, 'Kindly refer to the original image', (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            roi.append(cropped_img)
          else:
            cropped_img = cv2.copyMakeBorder(cropped_img, (150 - h), (150 - h), (150 - w), (150 - w), cv2.BORDER_CONSTANT, value = (0,0,0))
            cropped_img = cv2.putText(cropped_img, 'Too small to be displayed accurately', (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cropped_img = cv2.putText(cropped_img, 'Kindly refer to the original image', (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            roi.append(cropped_img)
    return roi

def resized_roi(img = img, roi = roi):
    """Here we resize the roi to 100*100"""
    resized_roi = []
    for cropped_img in roi:
      resized_cropped_img = cv2.resize(cropped_img, (100,100))
      resized_cropped_img = cv2.rectangle(resized_cropped_img, (0,0), (100,100), (0,0,0), 3)
      resized_roi.append(resized_cropped_img)
    return resized_roi


def number_of_detections(img, detections):
    """Here we find the number of objects detected"""
    if len(detections) == 1:
      return '1 object detected'
    else:
      return '{} objects detected'.format(len(detections))

def unique_items_detected(img, detected_objects):
    """Here we list all the unique objects detected"""
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
    """Here we find the number of occurences per unique object in the image"""
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
      save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
      if save_fig == 'y':
        img = df.plot.barh(x = 'Unique Item', y = 'Count')
        img = plt.suptitle('Count of each unique object detected', size = 15)
        img = input('Save image as ___.jpg ')
        img = plt.savefig(img + '.jpg')
        img = plt.close()
      else:
        pass
      return a
      

def detected_objects_shape(img = img, detected_objects = detected_objects, roi = roi):
    """Here we determine the shape of the detected object"""
    if len(detected_objects) == 0:
      return '0 objects detected'
    else:
      roi_shape = []
      for i in roi:
        roi_shape.append(i.shape)
      df = pd.DataFrame(columns = ['Object', 'Shape'])
      df['Object'] = detected_objects
      df['Shape'] = roi_shape
      print(df)
      save_df = input('Do you want to save this as a csv? Enter "y" for "Yes" and "n" for "No". ')
      if save_df == 'y':
        csv = input('Save as ___.csv ')
        csv = df.to_csv(csv + '.csv')
      else:
        pass
      #return df
 
def display_all_objects(img = img, detected_objects = detected_objects, roi = roi):
    """Here we display all the objects that have been detected"""
    if len(detected_objects) == 0:
      return '0 objects detected'
    elif len(detected_objects) == 1:
      a = plt.imshow(roi[0])
      a = plt.suptitle('The only {} detected in the image'.format(detected_objects[0]))
      a = plt.axis('off')
      a = plt.show()
      save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
      if save_fig == 'y':
        img = plt.imshow(roi[0])
        img = plt.axis('off')
        img = plt.suptitle('The only {} detected in the image'.format(detected_objects[0]))
        img = input('Save image as ___.jpg ')
        img = plt.savefig(img + '.jpg')
        img = plt.close()
      else:
        pass
      return a
    else:
      fig, ax = plt.subplots(1, len(detected_objects))
      for i in range(len(detected_objects)):
        a = ax[i].imshow(roi[i])
        a = ax[i].set_title(detected_objects[i])
        a = ax[i].axis('off')
      #a = fig.text(.5, .05, 'All objects detected in the image', ha = 'center')
      a = plt.show()
      save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
      if save_fig == 'y':
        fig, ax = plt.subplots(1, len(detected_objects))
        for i in range(len(detected_objects)):
          img = ax[i].imshow(roi[i])
          img = ax[i].set_title(detected_objects[i])
          img = ax[i].axis('off')
        #img = fig.suptitle('All ' + specific_object + 's detected in the image')
        img = input('Save image as ___.jpg ')
        img = plt.savefig(img + '.jpg')
        img = plt.close()
      else:
        pass
      return a

def display_specific_image(img = img, detected_objects = detected_objects, roi = roi):
    """Here we give an option to display a specific object detected in the image"""
    if len(detected_objects) == 0:
      return '0 objects detected'
    elif len(detected_objects) == 1:
      a = plt.imshow(roi[0])
      a = plt.suptitle('The only ' + detected_objects[0] + ' detected in the image')
      a = plt.axis('off')
      a = plt.show()
      save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
      if save_fig == 'y':
        img = plt.imshow(roi[0])
        img = plt.axis('off')
        img = plt.suptitle('The only ' + detected_objects[0] + ' detected in the image')
        img = input('Save image as ___.jpg ')
        img = plt.savefig(img + '.jpg')
        img = plt.close()
      else:
        pass
      return a
    else:
      df = pd.DataFrame(columns = ['detected_objects'])
      df['detected_objects'] = detected_objects
      specific_object = input('Which of these do you wish to see? : {} '.format(df['detected_objects'].unique()))
      specific_roi = []
      if specific_object in detected_objects:
        specific_object_df = df.loc[df['detected_objects'] == specific_object]
        specific_index = list(specific_object_df.index)
        for i in specific_index:
          specific_roi.append(roi[i])
      else:
        return 'Incorrect value entered'
      
      if len(specific_roi) == 1:
        a = plt.imshow(specific_roi[0])
        a = plt.suptitle('The only ' + specific_object + ' detected in the image')
        a = plt.axis('off')
        a = plt.show()
        save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
        if save_fig == 'y':
          img = plt.imshow(specific_roi[0])
          img = plt.axis('off')
          img = plt.suptitle('The only ' + specific_object + ' detected in the image')
          img = input('Save image as ___.jpg ')
          img = plt.savefig(img + '.jpg')
          img = plt.close()
        else:
          pass
        return a
      else:
        fig, ax = plt.subplots(1, len(specific_roi))
        for i in range(len(specific_roi)):
          a = ax[i].imshow(specific_roi[i])
          a = ax[i].axis('off')
        a = fig.suptitle('All ' + specific_object + 's detected in the image')
        a = plt.show()
        save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
        if save_fig == 'y':
          fig, ax = plt.subplots(1, len(specific_roi))
          for i in range(len(specific_roi)):
            img = ax[i].imshow(specific_roi[i])
            img = ax[i].axis('off')
          img = fig.suptitle('All ' + specific_object + 's detected in the image')
          img = input('Save image as ___.jpg ')
          img = plt.savefig(img + '.jpg')
          img = plt.close()
        else:
          pass
        return a


    
def display_all_resized_objects(img = img, detected_objects = detected_objects, resized_roi = resized_roi):
    """Here we display all the resized objects"""
    if len(detected_objects) == 0:
      return '0 objects detected'
    elif len(detected_objects) == 1:
      a = plt.imshow(resized_roi[0])
      a = plt.suptitle('Resized image of the only detected {}'.format(detected_objects[0]))
      a = plt.axis('off')
      a = plt.show()
      save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
      if save_fig == 'y':
        img = plt.imshow(resized_roi[0])
        img = plt.axis('off')
        img = plt.suptitle('Resized image of the only detected {}'.format(detected_objects[0]))
        img = input('Save image as ___.jpg ')
        img = plt.savefig(img + '.jpg')
        img = plt.close()
      else:
        pass
      return a
    else:
      resized_roi = np.hstack(resized_roi)
      a = plt.imshow(resized_roi)
      a = plt.suptitle('All resized images in the following order - {}'.format(detected_objects))
      a = plt.axis('off')
      a = plt.show()
      save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
      if save_fig == 'y':
        img = plt.imshow(resized_roi)
        img = plt.axis('off')
        img = plt.suptitle('All resized images in the following order - {}'.format(detected_objects))
        img = input('Save image as ___.jpg ')
        img = plt.savefig(img + '.jpg')
        img = plt.close()
      else:
        pass
      return a


def group_objects_by_class(img = img, detected_objects = detected_objects, resized_roi = resized_roi):
    """Here we display objects grouped by their classes"""
    if len(detected_objects) == 0:
      return '0 objects detected'
    elif len(detected_objects) == 1:
      a = plt.imshow(resized_roi[0])
      a = plt.axis('off')
      a = plt.suptitle('The only {} detected in the image'.format(detected_objects[0]))
      a = plt.show()
      save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
      if save_fig == 'y':
        img = plt.imshow(resized_roi[0])
        img = plt.axis('off')
        img = plt.suptitle('The only {} detected in the image'.format(detected_objects[0]))
        img = input('Save image as ___.jpg ')
        img = plt.savefig(img + '.jpg')
        img = plt.close()
      else:
        pass
      return a
    else:
      df = pd.DataFrame(columns = ['detected_objects'])
      df['detected_objects'] = detected_objects
      df_index = list(df.index)
      df['df_index'] = df_index
      df = df.groupby('detected_objects')['df_index'].apply(list)
      df = pd.DataFrame(df)
      unique_objects = list(df.index)
      unique_roi = []
      for i in df['df_index']:
        grouped_index = []
        grouped_roi = []
        for j in i:
          grouped_index.append(j)
        for k in grouped_index:
          grouped_roi.append(resized_roi[k])
        if len(grouped_roi) > 1:
          grouped_roi = np.hstack(grouped_roi)
        else:
          grouped_roi = grouped_roi[0]
        unique_roi.append(grouped_roi)
      if len(unique_roi) > 1:
        fig, ax = plt.subplots(len(unique_roi), 1)
        for i in range(len(unique_roi)):
          a = ax[i].imshow(unique_roi[i])
          a = ax[i].set_title(unique_objects[i], size = 11)
          a = ax[i].axis('off')
        a = fig.suptitle('Objects grouped by their respective classes : \n ',size = 15)
        a = plt.show()
        save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
        if save_fig == 'y':
          fig, ax = plt.subplots(len(unique_roi), 1)
          for i in range(len(unique_roi)):
            img = ax[i].imshow(unique_roi[i])
            img = ax[i].set_title(unique_objects[i], size = 11)
            img = ax[i].axis('off')
          img = fig.suptitle('Objects grouped by their respective classes : \n ',size = 15)
          img = input('Save image as ___.jpg ')
          img = plt.savefig(img + '.jpg')
          img = plt.close()
        else:
          pass
        return a
      else:
        a = plt.imshow(unique_roi[0])
        a = plt.axis('off')
        a = plt.suptitle('All {}s stacked together'.format(unique_objects[0]))
        a = plt.show()
        save_fig = input('Do you want to save this image? Enter "y" for "Yes" and "n" for "No". ')
        if save_fig == 'y':
          img = plt.imshow(unique_roi[0])
          img = plt.axis('off')
          img = plt.suptitle('All {}s stacked together'.format(unique_objects[0]))
          img = input('Save image as ___.jpg ')
          img = plt.savefig(img + '.jpg')
          img = plt.close()
        else:
          pass
        return a
