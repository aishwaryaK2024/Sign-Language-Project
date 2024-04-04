import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Change maxHands to 2 for detecting two hands

offset = 20
imSize = 300
counter = 0

folder = "Data/F"

print("Starting capture in 5 seconds...")
start_time = time.time()
while (time.time() - start_time) < 5:
    success, img = cap.read()
    cv2.imshow("Waiting...", img)
    cv2.waitKey(1)

print("Starting capture...")

while counter < 100:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # Initialize variables to store the coordinates of the bounding box
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for hand in hands:  # Loop through detected hands
            x, y, w, h = hand['bbox']
            # Update the bounding box coordinates
            x_min = min(x_min, x - offset)
            y_min = min(y_min, y - offset)
            x_max = max(x_max, x + w + offset)
            y_max = max(y_max, y + h + offset)

        # Crop the image using the combined bounding box
        imgCrop = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        imgWhite = np.ones((imSize, imSize, 3), np.uint8) * 255

        # Calculate the aspect ratio of the cropped region
        h, w, _ = imgCrop.shape
        aspectRatio = h / w

        # Resize the cropped image to fit into the white image
        if aspectRatio > 1:
            k = imSize / h
            wCal = math.ceil(k * w)
            imgRsize = cv2.resize(imgCrop, (wCal, imSize))
            wgap = math.ceil((imSize - wCal) / 2)
            imgWhite[:, wgap:wCal + wgap] = imgRsize
        else:
            k = imSize / w
            hCal = math.ceil(k * h)
            imgRsize = cv2.resize(imgCrop, (imSize, hCal))
            hgap = math.ceil((imSize - hCal) / 2)
            imgWhite[hgap:hCal + hgap, :] = imgRsize

        cv2.imshow("ImageCrop", imgCrop)  # Display cropped image
        cv2.imshow("ImageWhite", imgWhite)  # Display processed image

        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

print("Maximum image count reached. Exiting.")
