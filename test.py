import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5","model/labels.txt")

offset = 20
imSize = 300
counter = 0
labels = ["A","B","C","D","H","I","J","K","L","M","N","O","P","Q","R","T","U","V","W","X","Y","Z","S","HELLO"]

#folder = "Data/A"

while True:
    success,img = cap.read()
    imgOutput = img.copy()
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
            print("imgRsize Before:", imSize)
            imgRsize = cv2.resize(imgCrop,(wCal,imSize))
            print("imgRsize After:", imgRsize)
            imgRsizeShape = imgRsize.shape
            wgap = math.ceil((imSize-wCal)/2)
            imgWhite[:, wgap:wCal+wgap] = imgRsize
            prediction , index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)


        else:
            k = imSize / h
            hCal = math.ceil(k * h)
            imgRsize = cv2.resize(imgCrop, (imSize, hCal))
            imgRsizeShape = imgRsize.shape
            hgap = math.ceil((imSize - hCal) / 2)
            imgWhite[hgap:hCal + hgap,:] = imgRsize
            prediction , index = classifier.getPrediction(imgWhite,draw=False)

        cv2.rectangle(imgOutput,(x - offset,y - offset-50),(x - offset + 200,y - offset-50+50),(255,0,255),cv2.FILLED)
        print("Index : ",index)

        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_PLAIN,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x - offset,y - offset),(x + w+offset,y + h+offset),(255,0,255),4)

        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)
