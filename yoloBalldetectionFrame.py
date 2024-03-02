import cv2
import torch
from ultralytics import YOLO

# Load the pretrained model
model = YOLO('yolov8x-oiv7.pt')

# Load the image
img = cv2.imread('frame.jpg')

# Perform inference
results = model(img)

# Get the bounding boxes
for result in results:
    # Iterate over each detected object
    for detection in result.boxes.data:
        box = detection[:4]  # Bounding box coordinates
        score = detection[4]  # Confidence score
        class_id = detection[5]  # Class ID

        # Check if the label is for a chair
        if model.names[int(class_id)] == 'Person':
            # Draw a red rectangle around the chair
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

# Display the image with the bounding box
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()