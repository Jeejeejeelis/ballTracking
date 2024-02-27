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

def drawCircles(frame1, frame2, circles):
    if circles is not None:
        print("Circles detected at")
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            print(f"Circle found at ({i[0]}, {i[1]}) with radius {i[2]}")
            center = (i[0], i[1])
            # circle center
            cv.circle(frame1, center, 1, (0, 100, 100), 3)
            cv.circle(frame2, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame1, center, radius, (255, 0, 255), 3)
            cv.circle(frame2, center, radius, (255, 0, 255), 3)
    

def main(argv):
    print("opencv version:")
    print(cv.__version__)
    src = openFile(argv)
    
    # This bilateral filter makes all the difference for tennis ball detection.
    src2 = cv.bilateralFilter(src, 15, 1000, 1000);
    
    grayFrame = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
    # Gaussian blur another option to medianBlur!
    # Blur used to reduce noise and avoid false circle detection!
    grayFrame= cv.medianBlur(grayFrame,5)

    
    rows = grayFrame.shape[0] #grayframe height resolution!
    columns = grayFrame.shape[1]#grayframe width resolution!

    #Check houghCircleTransform function for parameter info.
    dp=1
    min_dist = rows/8
    param1 = 100
    param2 = 30
    min_rad = 1
    max_rad = 30
    circles = houghCircleTransform(grayFrame, dp, min_dist, param1, param2, min_rad, max_rad)
    drawCircles(src, grayFrame, circles)

    #display detected circles!
    cv.imshow("detected circles", src)
    cv.imshow("grayFrame circles", grayFrame)
    cv.waitKey(0)

if __name__== '__main__':
    main(sys.argv[1:])