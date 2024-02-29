import sys
import cv2 as cv
import numpy as np
from mask import tennisballMask, courtLinesMask

def openFile():
     #import default image or specified file from terminal command
    default_file = 'calibrateHSV_noSinner.jpg'
    src = cv.imread(cv.samples.findFile(default_file), cv.IMREAD_COLOR)

    #check image loaded correctly
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    return src

def create_calibrationMask():
    default_file = 'calibrateHSV_noSinner.jpg'
    src = openFile()

    #Create mask frame
    min_hue =200
    min_saturation = 5
    min_value =65
    max_hue = 250
    max_saturation = 45
    max_value = 100
    mask = courtLinesMask(src, min_hue,min_saturation,min_value,max_hue,max_saturation,max_value)
    cv.imshow("lineMask", mask)
    #DELETE ME
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
    #DELETE ME
    return mask