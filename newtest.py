import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)
    classifier = Classifier("M/keras_model.h5", "M/labels.txt")

    offset = 20
    imSize = 300
    labels = ["A", "B", "C","D" ,"E","F","G","H","I","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","HELP"]

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            # Initialize variables to store the coordinates of the combined bounding box
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            # Combine bounding boxes for all hands
            for hand in hands:
                x, y, w, h = hand['bbox']
                x_min = min(x_min, x - offset)
                y_min = min(y_min, y - offset)
                x_max = max(x_max, x + w + offset)
                y_max = max(y_max, y + h + offset)

            # Draw rectangle around the combined bounding box
            cv2.rectangle(imgOutput, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 4)

            # Crop the combined hand image
            combined_hand = img[int(y_min):int(y_max), int(x_min):int(x_max)]
            imgWhite = np.ones((imSize, imSize, 3), np.uint8) * 255

            # Resize the combined hand image to fit into the white image
            h, w, _ = combined_hand.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imSize / h
                wCal = math.ceil(k * w)
                imgRsize = cv2.resize(combined_hand, (wCal, imSize))
                wgap = math.ceil((imSize - wCal) / 2)
                imgWhite[:, wgap:wCal + wgap] = imgRsize
            else:
                k = imSize / w
                hCal = math.ceil(k * h)
                imgRsize = cv2.resize(combined_hand, (imSize, hCal))
                hgap = math.ceil((imSize - hCal) / 2)
                imgWhite[hgap:hCal + hgap, :] = imgRsize

            # Classify the combined hand image
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

            # Display the label above the hands
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.7, 2)[0]
            text_x = int((x_min + x_max - text_size[0]) / 2)
            text_y = int(y_min - 30)

            # Draw background rectangle for label text
            cv2.rectangle(imgOutput, (text_x - 10, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 10, text_y + text_size[1] + 5), (255, 0, 255), cv2.FILLED)

            # Draw the label text
            cv2.putText(imgOutput, label, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 2)

        cv2.imshow("Image", imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
