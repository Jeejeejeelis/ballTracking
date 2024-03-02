import cv2 as cv
from color import Color
import numpy as np

def drawCircles(frame, circles,colors, color):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            print(f"Circle found at ({i[0]}, {i[1]}) with radius {i[2]}")
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1,colors.get(color), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, colors.get(color), 3)

def drawSquares(frame, centers, colors, color):
    if centers is not None:
        centers = np.uint16(np.around(centers))
        for i in centers:
            #print(f"Approximate circle found at ({i[0]}, {i[1]})")
            center = (i[0], i[1])
            # Define the size of the square
            size = 20  # Adjust this value as needed
            # Define the top-left and bottom-right points of the square
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            # Draw the square
            cv.rectangle(frame, top_left, bottom_right, colors.get(color), 2)

def drawLines(frame, lines,colors, color):
     if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(frame, (x1, y1), (x2, y2), colors.get(color), 2)