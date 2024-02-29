import cv2 as cv
import numpy as np

def houghCircleTransform(frame, dp, min_dist, param1, param2, min_rad, max_rad):
    #Apply Hough Circle Transform
    # grayFrame: Input image (grayscale).
    # circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
    # HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
    # dp = 1: The inverse ratio of resolution.
    # min_dist = gray.rows/16: Minimum distance between detected centers.
    # param_1 = 200: Upper threshold for the internal Canny edge detector.
    # param_2 = 100*: Threshold for center detection.
    # min_radius = 0: Minimum radius to be detected. If unknown, put zero as default.
    # max_radius = 0: Maximum radius to be detected. If unknown, put zero as default.
    
    circles = cv.HoughCircles(frame, cv.HOUGH_GRADIENT, 1, min_dist,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)
    return circles

def findApproxCirclesFromMask(frame, margin):
     # Find contours in the mask
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # List to store the centers of the circular contours
    centers = []

    for contour in contours:
        # Calculate the area of the contour
        area = cv.contourArea(contour)

        # Calculate the area of the minimum enclosing circle
        (x, y), radius = cv.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)

        # If the two areas are approximately equal, the contour is likely to be circular
        if np.isclose(area, circle_area, rtol=margin):
            # This contour is approximately circular!
            # Add its center to the list
            centers.append((int(x), int(y)))
    return centers