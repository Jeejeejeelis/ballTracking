import cv2 as cv
import numpy as np

def tennisballMask(src,min_hue,min_saturation,min_value,max_hue,max_saturation,max_value):
    # Convert BGR to HSV
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    # Testing larger range!
    min_hue = min_hue / 2
    min_saturation = min_saturation *2.55
    min_value = min_value * 2.55
    max_hue = max_hue / 2
    max_saturation = max_saturation *2.55
    max_value = max_value * 2.55

    # Define range for tennis ball color in HSV
    lower_green = np.array([min_hue, min_saturation, min_value])
    upper_green = np.array([max_hue, max_saturation, max_value])

    # Threshold the HSV image to get only green colors
    mask = cv.inRange(hsv, lower_green, upper_green)

    # Define the kernel size for morphological operations
    # kernel = np.ones((5,5),np.uint8)
    #use a disk structuring element instead! Try out different values! 4 was too much
    disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    # # Perform erosion
    # erosion = cv.erode(mask, kernel, iterations = 1)
    # # Perform dilation
    # dilation = cv.dilate(erosion, kernel, iterations = 1)

    # Perform morphological operations
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, disk)
    # Perform closing. Useful in closing small holes or dark spots within the object.
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, disk)
    # Blur the image
    #blurred = cv.GaussianBlur(closing, (5, 5), 0)

    return closing
def courtLinesMask(src,min_hue,min_saturation,min_value,max_hue,max_saturation,max_value):
    # Convert BGR to HSV
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    min_hue = min_hue / 2
    min_saturation = min_saturation *2.55
    min_value = min_value * 2.55
    max_hue = max_hue / 2
    max_saturation = max_saturation *2.55
    max_value = max_value * 2.55

    # Define range for tennis ball color in HSV
    lower_color = np.array([min_hue, min_saturation, min_value])
    upper_color = np.array([max_hue, max_saturation, max_value])

    # Threshold the HSV image to get only green colors
    mask = cv.inRange(hsv, lower_color, upper_color)

    return mask