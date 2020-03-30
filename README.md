This is a python package to analyze detections conducted across different types of documents. In the current version, it supports only objects detected in images.

This version is compatible with object detection performed by [ImageAI](https://github.com/OlafenwaMoses/ImageAI) using the pre-trained weights of [ResNet50](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5).

Download and place the aforementioned pre-trained weights in your local folder.

# Installation
Install this package with the following command:
```
pip install --upgrade pyrack
```
In order to detect objects and perform some analysis on them, import the package with the following command:
```
from pyrack import object as obj
```

# Functionalities
The following functions can be implemented with the current version\
1) ## Reading an image

This function reads and stores an image in the RGB color-space in the form of numpy arrays
```
img = obj.img(image_path)
```

2) ## Displaying the image

This loads the image that was read in the previous step

```
obj.display_original_image(img)
```

3) ## Defining the model path

Here, you need to mention the path where the pre-trained weights file are located in your system

```
model_path = obj.model_weights(os.path.join(path, 'resnet50_coco_best_v2.0.1.h5'))
```

4) ## Detecting objects in the image

This function detects all the objects present in the with the help of pre-trained weights of ResNet50

```
detections = obj.detections(img)
```

5) ## Get all detected objects

Here, you get to see all the objects that have been detected from the previous step

```
detected_objects = obj.detected_objects(img, detections)
print(detected_objects)
```

6) ## Getting the bounding-box(bbox) coordinates for all the detected objects

Here, we get the bbox co-ordinates within which the objects are detected

```
bbox = obj.bbox(detections)
```

7) ## Getting the regions of interest(roi)

From the bbox co-ordinates extracted above, we can draw the regions of interest around each detected object

```
roi = obj.roi(img, detections)
```

8) ## Resizing the roi's

Here, we can resize each roi to a dimension of 100*100*3. Some objects which are too small to be resized are first padded with black pixels and are then resized.

```
resized_roi = obj.resized_roi(img, roi)
```

9) ## Number of detections

This function returns the total number of objects that have been detected in the image

```
obj.number_of_detections(img, detections)
```

10) ## Unique items detected

This function returns all the unique items detected in the image

```
obj.unique_items_detected(img, detected_objects)
```

11) ## Count of each unique object

This returns a plot of the count of each unique object if the number of detections is greater than 1. You also have an option to save this plot in your system.

```
obj.count_per_unique_item(img, detected_objects)
```

12) ## Shape of each detected object

This returns the 3-dimensional shape of each detected object. You also have an option to save it as a .csv file in your system.

```
obj.detected_objects_shape(img, detected_objects, roi)
```

13) ## Display all detected objects

Here, you get to see an image of all the detected objects. You also have an option to save this as a .jpg file.

```
obj.display_all_objects(img, detected_objects, roi)
```

14) ## Display specific objects

Using this function, you have the option of viewing and saving images of specific objects detected in the image.

```
obj.display_specific_image(img, detected_objects, roi)
```

15) ## Display all resized objects

Here, you get to see and save resized images of all objects that were detected.

```
obj.display_all_resized_objects(img, detected_objects, resized_roi)
```

16) ## Grouping objects by class

Here, you get to view objects grouped by classes. You also have the option to save it as a .jpg file.

```
obj.group_objects_by_class(img, detected_objects, resized_roi)
```




 