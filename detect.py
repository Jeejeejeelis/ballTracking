import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

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
    circles = []

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
            circles.append((int(x), int(y), radius))
    return circles

# def average_lines(lines, frame):
#     if lines is None:
#         return None

#     # Calculate the slopes and intercepts of the lines
#     slopes = [(y2 - y1) / (x2 - x1) if (x2-x1) else 0 for x1, y1, x2, y2 in lines[:, 0]]
#     intercepts = [y1 - slope * x1 for (x1, y1, _, _), slope in zip(lines[:, 0], slopes)]

#     # Average the slopes and intercepts
#     avg_slope = sum(slopes) / len(slopes) if slopes else 0
#     avg_intercept = sum(intercepts) / len(intercepts) if intercepts else 0

#     # Calculate the start and end points of the average line
#     x1 = 0
#     y1 = int(avg_intercept)
#     x2 = frame.shape[1]
#     y2 = int(avg_slope * x2 + avg_intercept)

#     return np.array([[x1, y1, x2, y2]])


def findLines(frame):
    # Apply edge detection, If i find too many lines make second input 150 or more!
    edges = cv.Canny(frame, 20, 100)

    # Use Hough Line Transform to detect lines
    # The arguments are the binarized image, rho, theta and the threshold
    # rho: Distance resolution of the accumulator in pixels.
    # theta: Angle resolution of the accumulator in radians.
    # threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes (> `threshold`).
    #lines = cv.HoughLines(edges, 1, np.pi/180, 200)


    # edges: This is the output of an edge detection operation (like cv.Canny()). It’s a binary image where edges are white and the rest is black.
    # 1: This is the resolution of the parameter r in pixels. r is the distance resolution from the origin to the detected line in the accumulator.
    # np.pi/180: This is the resolution of the parameter θ in radians. θ is the angle resolution of the detected line in the accumulator.
    # 100: This is the threshold parameter. Only those lines are returned that get enough votes (>100).
    # minLineLength=100: This is the minimum line length. Line segments shorter than this are rejected.
    # maxLineGap=10: This is the maximum allowed gap between line segments lying on the same line to treat them as a single line.
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 75, minLineLength=100, maxLineGap=50)
    return lines

def yoloDetect(model, results):
    detectedObjects = []  #Store detected objects

    for result in results:
        for detection in result.boxes.data:
            box = detection[:4]  # Bounding box coordinates
            score = detection[4]  # Confidence score
            class_id = detection[5]  # Class ID
            # Check if the label is for a tennis racket
            if model.names[int(class_id)] == 'Tennis racket':
                # Add the bounding box to the list
                detectedObjects.append(box)

    return detectedObjects