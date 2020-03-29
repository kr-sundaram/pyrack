import cv2
import matplotlib.pyplot as plt

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