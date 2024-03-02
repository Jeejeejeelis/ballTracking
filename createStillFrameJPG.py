import cv2

# Open the video
cap = cv2.VideoCapture('sinner_2560×1440.mp4')

# Check if video opened successfully
if not cap.isOpened(): 
    print("Error opening video")

# Idea is to capture the 300th frame and save it as .jpg 
#Loop the first 300 frames
whichFrame = 225
for i in range(whichFrame):
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        break
     # Save the 200th frame as an image
    if i == whichFrame-1:
        cv2.imwrite('frame.jpg', frame)

# Release the video capture object
cap.release()