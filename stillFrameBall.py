import sys
import cv2 as cv
import numpy as np

def openFile(argv):
     #import default image or specified file from terminal command
    default_file = 'frame.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    #check image loaded correctly
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    return src

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
                               minRadius=1, maxRadius=24)
    return circles

def drawCircles(frame, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            print(f"Circle found at ({i[0]}, {i[1]}) with radius {i[2]}")
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255), 3)

def drawSquares(frame, centers):
    if centers is not None:
        centers = np.uint16(np.around(centers))
        for i in centers:
            print(f"Approximate circle found at ({i[0]}, {i[1]})")
            center = (i[0], i[1])
            # Define the size of the square
            size = 20  # Adjust this value as needed
            # Define the top-left and bottom-right points of the square
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            # Draw the square
            cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

def tennisballMask(src):
    # Convert BGR to HSV
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    # #convert GIMP values to opencv values straight away!
    min_hue = 67.1 /2
    min_saturation = 11.3 * 2.55
    min_value = 23 * 2.55
    max_hue = 155 / 2
    max_saturation = 60 *2.55
    max_value = 100 * 2.55

    # min_hue = 67.1 / 2
    # min_saturation = 16.1 *2.55
    # min_value = 34.4 * 2.55
    # max_hue = 155 / 2
    # max_saturation = 60 *2.55
    # max_value = 100 * 2.55

    # Define range for tennis ball color in HSV
    lower_green = np.array([min_hue, min_saturation, min_value])
    upper_green = np.array([max_hue, max_saturation, max_value])

    # Threshold the HSV image to get only green colors
    mask = cv.inRange(hsv, lower_green, upper_green)

    # # Define the kernel size for morphological operations
    # # kernel = np.ones((5,5),np.uint8)
    # #use a disk structuring element instead! Try out different values! 4 was too much
    disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    
    # # # Perform erosion
    # # erosion = cv.erode(mask, kernel, iterations = 1)
    # # # Perform dilation
    # # dilation = cv.dilate(erosion, kernel, iterations = 1)

    # # Perform morphological operations
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, disk)
    # # Perform closing. Useful in closing small holes or dark spots within the object.
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, disk)

    # # Blur the image
    # #blurred = cv.GaussianBlur(closing, (5, 5), 0)

    #HoughCircles expects hollow ball so find edges of the mask!
    #minVal and maxVal for the hysteresis procedure.
    # If a pixel gradient value is below minVal, it is considered not to be an edge.
    # If the pixel gradient is between minVal and maxVal, it is accepted as an edge 
    # only if it is connected to a pixel with a gradient value more than maxVal

    #edges = cv.Canny(blurred, 50, 150)

    # Find contours in the edge image
    #contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Calculate the perimeter of each contour
    #perimeters = [cv.arcLength(contour, True) for contour in contours]

    # Draw a circle around each contour
    # for contour in contours:
    #     # Calculate the center and radius of the minimum enclosing circle
    #     (x, y), radius = cv.minEnclosingCircle(contour)
    #     center = (int(x), int(y))
    #     radius = int(radius)

    #     # Draw the circle on the source image
    #     cv.circle(src, center, radius, (0, 255, 0), 2)

    
    return closing

def findApproxCirclesFromMask(mask):
     # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # List to store the centers of the circular contours
    centers = []

    for contour in contours:
        # Calculate the area of the contour
        area = cv.contourArea(contour)

        # Calculate the area of the minimum enclosing circle
        (x, y), radius = cv.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)

        # If the two areas are approximately equal, the contour is likely to be circular
        if np.isclose(area, circle_area, rtol=0.25):
            # This contour is approximately circular!
            # Add its center to the list
            centers.append((int(x), int(y)))
    return centers

    
    

def main(argv):
    print("opencv version:")
    print(cv.__version__)
    src = openFile(argv)

    #lets make a mask of the denoised, contrasted and sharpened frame!
    #mask = tennisballMask(src)

    # Apply noise reduction
    denoised_frame = cv.fastNlMeansDenoisingColored(src, None, 10, 10, 7, 21)

    # Increase contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    contrasted_frame = cv.convertScaleAbs(denoised_frame, alpha=alpha, beta=beta)

    
    # This bilateral filter makes all the difference for tennis ball detection.
    # src: Source image.
    # d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
    # sigmaColor: Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood
    # will be mixed together, resulting in larger areas of semi-equal color.
    # sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other
    # as long as their colors are close enough. When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
    # src2 = cv.bilateralFilter(src, 15, 1000, 1000); # original
    src2 = cv.bilateralFilter(contrasted_frame, 15, 1000, 1000); # test better values
    
    # Define the sharpening kernel
    # kernel = np.array([[0, -1, 0],
    #                [-1, 5,-1],
    #                [0, -1, 0]])
    
    kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
    

    # Apply the kernel to the grayscale and mask image using the filter2D function
    sharpened = cv.filter2D(src2, -1, kernel)

    #Create mask frame
    mask = tennisballMask(src)

    #create a gray frame
    grayFrame = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)

   
    # Gaussian blur another option to medianBlur!
    # Blur used to reduce noise and avoid false circle detection!
    #grayFrame= cv.medianBlur(sharpenedGray,5)

    #Testing gaussianBlur omstead of median blur to preserve edges
    #(width, height) Both numbers should be positive and odd. 
    blurred_grayFrame = cv.GaussianBlur(grayFrame, (5, 5), 0)

    rows = grayFrame.shape[0] #grayframe height resolution!
    columns = grayFrame.shape[1]#grayframe width resolution!

    #Check houghCircleTransform function for parameter info.
    # dp=1 # original
    dp=1
    min_dist = rows/8 # original
    # param1 = 100 # original. decrease if ball not detected
    param1 = 100
    # param2 = 30 #original. smaller number, more false positives
    param2 = 30
    min_rad = 1
    max_rad = 30
    circlesGrayFrame = houghCircleTransform(blurred_grayFrame, dp, min_dist, param1, param2, min_rad, max_rad)
    circlesMask = houghCircleTransform(mask, dp, min_dist, param1, param2, min_rad, max_rad)
    approxCirclesMask = findApproxCirclesFromMask(mask)

    drawSquares(src, approxCirclesMask)
    print("grayFrame circles: ")
    drawCircles(grayFrame,  circlesGrayFrame)
    print("Mask circles: ")
    # i cant see the circles in the mask so why bother!
    drawCircles(mask, circlesMask)
    ## Use code below to draw circles into src frame.
    drawCircles(src,circlesMask)
    # drawCircles(src,circlesGrayFrame)


    #display detected circles!
    cv.imshow("detected circles", src)
    cv.imshow("grayFrame circles", grayFrame)
    cv.imshow("mask", mask)
    cv.waitKey(0)

if __name__== '__main__':
    main(sys.argv[1:])