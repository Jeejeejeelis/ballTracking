import cv2
import numpy as np

def findCircles(frame, p1, p2, minRadius, maxRadius):
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1,100, p1, p2, minRadius, maxRadius)
    return circles

def main():
    # Read the image
    frame = cv2.imread('frame.jpg')
    prevCircle = None

    #square of the distance betweem two points of a frame
    dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2
    #from BGR to gray colorspace
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # use gaussian blur!
    #blurFrame = cv2.GaussianBlur(grayFrame, (17,17), 0)
    blurFrame = cv2.medianBlur(grayFrame, 5)
    # if there is a loop!
    #if not ret: break

    # #convert GIMP values to opencv values straight away!
    hue = 84.7 / 2
    saturation = 58.7 *2.55
    value = 120.1 * 2.55


    

    # Define range for tennis ball color in HSV
    rangeModifier = 46
    lower_green = np.array([hue - rangeModifier, saturation - (4*rangeModifier), value - (4*rangeModifier)])
    upper_green = np.array([hue + rangeModifier, saturation + (4*rangeModifier), value + (4*rangeModifier)])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    
    #try from 1 to 1.4 for last value! 100 is distance between two circles! param1 is sensitivity to find circles! param2 is accuracy(how many edgepoints)

    param1 = 100
    param2 = 30
    minRadius = 1
    maxRadius = 30

    
    # testValue = 10
    circles = findCircles(blurFrame, param1, param2, minRadius,maxRadius)
    # for i in range(5):
    #     param1 = param1 - testValue 
    #     param2 = param2 + testValue
    #     circles = findCircles(mask, param1, param2, minRadius,maxRadius)
    #     if circles is not None:
    #         circles = np.uint16(np.around(circles))
    #         print("We found", len(circles[0]), "circles with params:", param1, param2)

    # circles = findCircles(mask, param1, param2, minRadius,maxRadius)
    # #draw all circles found:
    # if circles is not None:
    #     print("Circles found!")
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         # print circle x_loc,y_loc and radius to terminal!
    #         print(f"Circle found at ({i[0]}, {i[1]}) with radius {i[2]}")
    #         # Draw the outer circle
    #         cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # Draw the center of the circle
    #         cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    # else:
    #     print("No circles found")


    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1],prevCircle[0], prevCircle[1]) < dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen = i
        cv2.circle(frame,(chosen[0], chosen[1]), 1, (0,100,100), 1)
        cv2.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,0,255), 1)
        prevCircle = chosen
    else:
        print("No circles found")

    cv2.imshow("circles", frame)
    cv2.imshow("mask", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__=="__main__":
        main()

