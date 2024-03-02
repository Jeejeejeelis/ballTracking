import cv2
import torch
from ultralytics import YOLO

# Load the pretrained model
model = YOLO('yolov8x-oiv7.pt')

# Open the video file
cap = cv2.VideoCapture('sinner_2560×1440.mp4')

while(cap.isOpened()):
    # Read the frame
    ret, frame = cap.read()
    if ret == True:
        # Perform inference
        results = model(frame)

        # Get the bounding boxes
        for result in results:
            # Iterate over each detected object
            for detection in result.boxes.data:
                box = detection[:4]  # Bounding box coordinates
                score = detection[4]  # Confidence score
                class_id = detection[5]  # Class ID

                # Check if the label is for a tennis ball
                if model.names[int(class_id)] == 'Tennis racket':
                    # Draw a red rectangle around the tennis ball
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        # Display the frame with the bounding box
        cv2.imshow('Frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()