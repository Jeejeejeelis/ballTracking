import sys
import cv2 as cv
import numpy as np
from color import Color
from draw import drawCircles, drawRectangles, drawLines
from detect import houghCircleTransform, findApproxCirclesFromMask, findLines, yoloDetect
from mask import tennisballMask, courtLinesMask
from calibrationMask import *
import torch
from ultralytics import YOLO


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
    # Load the pretrained model
    model = YOLO('yolov8x-oiv7.pt')
    # Open the video
    src = openFile(argv)
    fps = src.get(cv.CAP_PROP_FPS)

    #Save edited frames as mp4!
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, fps, (2560, 1440))

    color_instance = Color()

    # create mask of lines from our calibration image.
    #I am not getting nice results with the lines.
    # lineMask = create_calibrationMask()
    # courtLines = findLines(lineMask)

    # Set the starting point to 3 seconds
    start_frame = int(fps * 3)  # 3 seconds
    src.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    # Calculate the ending point (30 seconds)
    end_frame = int(fps * 13)  # 30 seconds now testing 4 sec

    current_frame = start_frame
    
    while(src.isOpened()):
        ret, frame = src.read()
        if ret:
            #print current frame!
            print(f"Processing frame: {src.get(cv.CAP_PROP_POS_FRAMES)}")
            #use YOLOv8 to detect objects in model
            results = model(frame)
            detectedObjects=yoloDetect(model, results)
            
            # # Apply noise reduction
            # denoised_frame = cv.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # Increase contrast
            # alpha = 1.5  # Contrast control (1.0-3.0)
            # beta = 0  # Brightness control (0-100)
            # contrasted_frame = cv.convertScaleAbs(denoised_frame, alpha=alpha, beta=beta)

            # This bilateral filter makes all the difference for tennis ball detection in grayscale.
            # src: Source image.
            # d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
            # sigmaColor: Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood
            # will be mixed together, resulting in larger areas of semi-equal color.
            # sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other
            # as long as their colors are close enough. When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
            # src2 = cv.bilateralFilter(src, 15, 1000, 1000); # original
            # frame2 = cv.bilateralFilter(contrasted_frame, 15, 1000, 1000)
            
            # kernel = np.array([[-1, -1, -1],
            #             [-1,  9, -1],
            #             [-1, -1, -1]])
            
            # # Apply the kernel to the grayscale and mask image using the filter2D function
            # sharpened = cv.filter2D(frame2, -1, kernel)
            
            #Create mask based on tennisball HSV values. Use original frame, filtering made it worse.
            # # Testing larger range!
            min_hue = 50
            min_saturation = 10
            min_value = 20
            max_hue = 170
            max_saturation = 100
            max_value = 100
            ballMask = tennisballMask(frame, min_hue, min_saturation, min_value,max_hue,max_saturation,max_value)
            
            # #create a gray frame. Try with sharpened aswell!
            #grayFrame = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)

            # #Testing gaussianBlur omstead of median blur to preserve edges
            # #(width, height) Both numbers should be positive and odd. 
            #blurred_grayFrame = cv.GaussianBlur(grayFrame, (3, 3), 0)

            # rows = grayFrame.shape[0] #grayframe height resolution!
            # columns = grayFrame.shape[1]#grayframe width resolution!

            #Check houghCircleTransform function for parameter info.
            # dp=1
            # min_dist = rows/8
            # param1 = 100
            # param2 = 30
            # min_rad = 1
            # max_rad = 30
            # circlesMask = houghCircleTransform(ballMask, dp, min_dist, param1, param2, min_rad, max_rad)
            approxCirclesMask = findApproxCirclesFromMask(ballMask, 0.25)
            #circlesGrayFrame = houghCircleTransform(blurred_grayFrame, dp, min_dist, param1, param2, min_rad, max_rad)
            #approxCirclesGrayFrame = findApproxCirclesFromMask(ballMask, 0.25)
            #drawCircles(frame, circlesGrayFrame,color_instance,"gray")
            drawRectangles(frame, detectedObjects, color_instance, "red")
            drawCircles(frame, approxCirclesMask,color_instance,"green")
            #drawRectangles(frame, approxCirclesMask,color_instance,"green")
            # The lines are not returning nice result...
            #drawLines(frame, courtLines, color_instance, "red")

            #Display edited videos in openCV.
            #cv.moveWindow("grayFrame", 0, 0)
            #cv.imshow("linemask", lineMask)
            #cv.imshow("ballmask", ballMask)
            #cv.moveWindow("ballmask", frame.shape[0], 0)
            #cv.imshow("grayFrame circles", grayFrame)

            #cv.imshow("original", frame)
            #write frames into new video file.
            out.write(frame)
            
            if cv.waitKey(1) & 0xFF == ord('q') or current_frame==end_frame:
                break
            current_frame = current_frame+1
        else:
            break

    src.release()
    out.release()
    cv.destroyAllWindows()

if __name__== '__main__':
    main(sys.argv[1:])