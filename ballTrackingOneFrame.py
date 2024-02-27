import cv2
import numpy as np

# Read the image
frame = cv2.imread('frame.jpg')


#GIMP ball color values
# red= 90.6
# green = 98.8
# blue = 67.8
#convert GIMP values to opencv values straight away!
hue = 84.7 / 2
saturation = 58.7 *2.55
value = 120.1 * 2.55


# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define range for tennis ball color in HSV
rangeModifier = 45
lower_green = np.array([hue - rangeModifier, saturation - (4*rangeModifier), value - (4*rangeModifier)])
upper_green = np.array([hue + rangeModifier, saturation + (4*rangeModifier), value + (4*rangeModifier)])

# Threshold the HSV image to get only green colors
mask = cv2.inRange(hsv, lower_green, upper_green)

# #Morphological operations to smooth mask
# # Define a kernel for the morphological operations
# kernel = np.ones((5,5),np.uint8)
# # Use erosion followed by dilation (this is called opening)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#Apply Gaussian blur to mask
mask = cv2.GaussianBlur(mask, (15, 15), 0)

# Apply Hough Circle Transform to detect the ball
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, minRadius=28-5, maxRadius=28+5)

# # find dp where circles are being detected!
# for minDist in range(50, 150, 10):
#     for minRadius in range(5, 15, 1):
#         for maxRadius in range(20, 30, 1):
#             circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minDist, minRadius=minRadius, maxRadius=maxRadius)
#             if circles is not None:
#                 print(f"minDist={minDist}, minRadius={minRadius}, maxRadius={maxRadius}: {len(circles[0])} circles detected")

if circles is not None:
    print("Ball found!")
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Draw the bounding box
        cv2.rectangle(frame, (x-r, y-r), (x+r, y+r), (0, 0, 255), 2)

cv2.imshow('Original Frame', frame)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()