import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imSize = 300
counter = 0

folder = "Data/Z"

while True:
    success,img = cap.read()
    hands,img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']
        imgWhite = np.ones((imSize,imSize,3),np.uint8)*255

        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        #imgWhite[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imSize/h
            wCal = math.ceil(k*w)
            imgRsize = cv2.resize(imgCrop,(wCal,imSize))
            imgRsizeShape = imgRsize.shape
            wgap = math.ceil((imSize-wCal)/2)
            imgWhite[:, wgap:wCal+wgap] = imgRsize[:,:imSize]
        else:
            k = imSize / h
            hCal = math.ceil(k * h)
            imgRsize = cv2.resize(imgCrop, (imSize, hCal))
            imgRsizeShape = imgRsize.shape
            hgap = math.ceil((imSize - hCal) / 2)
            imgWhite[hgap:hCal + hgap,:] = imgRsize[:imSize,:]

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
