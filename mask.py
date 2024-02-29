import cv2 as cv
import numpy as np

def tennisballMask(src):
    # Convert BGR to HSV
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    # #convert GIMP values to opencv values straight away!
    min_hue = 67.1 / 2
    min_saturation = 16.1 *2.55
    min_value = 34.4 * 2.55
    max_hue = 155 / 2
    max_saturation = 60 *2.55
    max_value = 100 * 2.55

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