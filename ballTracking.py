import sys
import cv2 as cv
import numpy as np
from color import Color
from draw import drawCircles, drawSquares, drawLines
from detect import houghCircleTransform, findApproxCirclesFromMask, findLines
from mask import tennisballMask, courtLinesMask
from calibrationMask import *

def openFile(argv):
    #import default video or specified file from terminal command
    default_file = 'sinner_2560×1440.mp4'
    filename = argv[0] if len(argv) > 0 else default_file
    # Open the video file
    src = cv.VideoCapture(filename)

    #check video loaded correctly
    if not src.isOpened():
        print ('Error opening video!')
        print ('Usage: hough_circle.py [video_name -- default ' + default_file + '] \n')
        return -1
    else:
        fps = src.get(cv.CAP_PROP_FPS)
        print("Frame rate of the video: ", fps)
        resolution_width = src.get(cv.CAP_PROP_FRAME_WIDTH)
        resolution_height = src.get(cv.CAP_PROP_FRAME_HEIGHT)
        print(f"The resolution of the video is {int(resolution_width)}x{int(resolution_height)} pixels.")
    
    return src
           

def main(argv):
    # Open the video
    src = openFile(argv)
    color_instance = Color()

    # create mask of lines from our calibration image.
    #I am not getting nice results with the lines.
    # lineMask = create_calibrationMask()
    # courtLines = findLines(lineMask)

    # Set the starting point to 3 seconds
    fps = src.get(cv.CAP_PROP_FPS)
    start_frame = int(fps * 3)  # 3 seconds
    src.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    # Calculate the ending point (30 seconds)
    end_frame = int(fps * 30)  # 30 seconds

    current_frame = start_frame

    #convert GIMP values to opencv values straight away!
    # Define range for tennis ball color in HSV
    
    while(src.isOpened()):
        ret, frame = src.read()
        if ret:
            #print current frame!
            print(f"Processing frame: {src.get(cv.CAP_PROP_POS_FRAMES)}")
            # Apply noise reduction
            denoised_frame = cv.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # Increase contrast
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 0  # Brightness control (0-100)
            contrasted_frame = cv.convertScaleAbs(denoised_frame, alpha=alpha, beta=beta)

            # This bilateral filter makes all the difference for tennis ball detection in grayscale.
            # src: Source image.
            # d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
            # sigmaColor: Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood
            # will be mixed together, resulting in larger areas of semi-equal color.
            # sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other
            # as long as their colors are close enough. When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
            # src2 = cv.bilateralFilter(src, 15, 1000, 1000); # original
            frame2 = cv.bilateralFilter(contrasted_frame, 15, 1000, 1000)
            
            kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
            
            # Apply the kernel to the grayscale and mask image using the filter2D function
            sharpened = cv.filter2D(frame2, -1, kernel)
            
            #Create mask based on tennisball HSV values. Use original frame, filtering made it worse.
            # # Testing larger range!
            min_hue = 50
            min_saturation = 10
            min_value = 20
            max_hue = 170
            max_saturation = 100
            max_value = 100
            ballMask = tennisballMask(frame, min_hue, min_saturation, min_value,max_hue,max_saturation,max_value)
            
            
            #create a gray frame. Try with sharpened aswell!
            grayFrame = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)

            #Testing gaussianBlur omstead of median blur to preserve edges
            #(width, height) Both numbers should be positive and odd. 
            blurred_grayFrame = cv.GaussianBlur(grayFrame, (3, 3), 0)

            rows = grayFrame.shape[0] #grayframe height resolution!
            columns = grayFrame.shape[1]#grayframe width resolution!

            #Check houghCircleTransform function for parameter info.
            dp=1
            min_dist = rows/8
            param1 = 100
            param2 = 30
            min_rad = 1
            max_rad = 30
            circlesMask = houghCircleTransform(ballMask, dp, min_dist, param1, param2, min_rad, max_rad)
            approxCirclesMask = findApproxCirclesFromMask(ballMask, 0.25)
            circlesGrayFrame = houghCircleTransform(blurred_grayFrame, dp, min_dist, param1, param2, min_rad, max_rad)
            approxCirclesGrayFrame = findApproxCirclesFromMask(blurred_grayFrame, 0.25)
            drawCircles(frame, circlesGrayFrame,color_instance,"gray")
            drawSquares(frame, approxCirclesGrayFrame,color_instance,"gray")
            drawCircles(frame, circlesMask,color_instance,"green")
            drawSquares(frame, approxCirclesMask,color_instance,"green")
            # The lines are not returning nice result...
            #drawLines(frame, courtLines, color_instance, "red")

            #display detected circles!
            cv.imshow("original", frame)
            #cv.moveWindow("grayFrame", 0, 0)
            cv.imshow("linemask", lineMask)
            cv.imshow("ballmask", ballMask)
            cv.moveWindow("ballmask", frame.shape[0], 0)
            #cv.imshow("grayFrame circles", grayFrame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    src.release()
    cv.destroyAllWindows()

if __name__== '__main__':
    main(sys.argv[1:])