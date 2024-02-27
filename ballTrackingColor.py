import cv2
import numpy as np

# Open the video
cap = cv2.VideoCapture('frame.jpg')
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate of the video: ", fps)
resolution_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
resolution_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"The resolution of the video is {int(resolution_width)}x{int(resolution_height)} pixels.")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for tennis ball color in HSV
        lower_green = np.array([29, 86, 6])
        upper_green = np.array([64, 255, 255])

        # #Draw big red box to check that i can draw on the video!
        # # Define the box dimensions
        # start_x, start_y = int(resolution_width * 0.25), int(resolution_height * 0.25)
        # end_x, end_y = int(resolution_width * 0.75), int(resolution_height * 0.75)

        # # Draw the red box
        # cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
        # #End of draw test!
        # This works!

        # Test detection! Ball detected should appear in terminal!

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply Hough Circle Transform to detect the ball
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100)

        if circles is not None:
            #check if a ball is detected!
            print("Ball found")
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                #cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x-r, y-r), (x+r, y+r), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()