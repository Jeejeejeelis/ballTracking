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
                               minRadius=1, maxRadius=30)
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
    kernel = np.ones((5,5),np.uint8)

    # # Perform erosion
    # erosion = cv.erode(mask, kernel, iterations = 1)
    # # Perform dilation
    # dilation = cv.dilate(erosion, kernel, iterations = 1)

    # Perform morphological operations
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # Perform closing. Useful in closing small holes or dark spots within the object.
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # Blur the image
    blurred = cv.GaussianBlur(closing, (5, 5), 0)

    return blurred
    

def main(argv):
    print("opencv version:")
    print(cv.__version__)
    src = openFile(argv)

    mask = tennisballMask(src)

    # Increase contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    contrasted_frame = cv.convertScaleAbs(src, alpha=alpha, beta=beta)

    # Apply noise reduction
    denoised_frame = cv.fastNlMeansDenoisingColored(contrasted_frame, None, 10, 10, 7, 21)
    
    # This bilateral filter makes all the difference for tennis ball detection.
    # src: Source image.
    # d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
    # sigmaColor: Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood
    # will be mixed together, resulting in larger areas of semi-equal color.
    # sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other
    # as long as their colors are close enough. When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
    # src2 = cv.bilateralFilter(src, 15, 1000, 1000); # original
    src2 = cv.bilateralFilter(denoised_frame, 15, 1000, 1000); # test better values

    

    # Convert BGR to HSV
    hsv = cv.cvtColor(src2, cv.COLOR_BGR2HSV)

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


    #create a gray frame
    grayFrame = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
    
    # Define the sharpening kernel
    # kernel = np.array([[0, -1, 0],
    #                [-1, 5,-1],
    #                [0, -1, 0]])
    
    kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
    

    # Apply the kernel to the grayscale and mask image using the filter2D function
    sharpenedGray = cv.filter2D(grayFrame, -1, kernel)
    sharpenedMask = cv.filter2D(mask, -1, kernel)

   
    # Gaussian blur another option to medianBlur!
    # Blur used to reduce noise and avoid false circle detection!
    #grayFrame= cv.medianBlur(sharpenedGray,5)

    #Testing gaussianBlur omstead of median blur to preserve edges
    #(width, height) Both numbers should be positive and odd. 
    grayFrame = cv.GaussianBlur(sharpenedGray, (5, 5), 0)
    mask= cv.GaussianBlur(sharpenedMask, (5, 5), 0)

    
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
    circlesGrayFrame = houghCircleTransform(grayFrame, dp, min_dist, param1, param2, min_rad, max_rad)
    circlesMask = houghCircleTransform(mask, dp, min_dist, param1, param2, min_rad, max_rad)
    
    print("grayFrame circles: ")
    drawCircles(grayFrame,  circlesGrayFrame)
    print("Mask circles: ")
    drawCircles(mask, circlesMask)
    ## Use code below to draw circles into src frame.
    # drawCircles(src,circlesMask)
    # drawCircles(src,circlesGrayFrame)

    #display detected circles!
    cv.imshow("detected circles", src)
    cv.imshow("grayFrame circles", grayFrame)
    cv.imshow("mask", mask)
    cv.waitKey(0)

if __name__== '__main__':
    main(sys.argv[1:])