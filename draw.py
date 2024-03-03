import cv2 as cv
from color import Color
import numpy as np

def drawCircles(frame, circles, colors, color):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles:
            print(f"Circle found at ({i[0]}, {i[1]}) with radius {i[2]}")
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, colors.get(color), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, colors.get(color), 3)

def drawRectangles(frame, boxes, colors, color):
    if boxes is not None:
        for box in boxes:
            # Draw a rectangle around the detected object
            cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors.get(color), 2)


def drawLines(frame, lines, colors, color):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(frame, (x1, y1), (x2, y2), colors.get(color), 2)