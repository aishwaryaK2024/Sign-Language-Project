import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


class SignRecognitionWindow:
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("Sign Language Recognition")

        # Get screen width and height
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()

        # Set window size to match screen size
        self.parent.geometry(f"{screen_width}x{screen_height}+0+0")



        # OpenCV setup for webcam
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=2)
        self.classifier = Classifier("M/keras_model.h5", "M/labels.txt")

        self.offset = 20
        self.imSize = 300
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
                       "U", "V", "W", "X", "Y", "Z", "HELP"]

        # Create frame for holding sign recognition functionality
        self.recognition_frame = tk.Frame(parent, bg="cyan")
        self.recognition_frame.place(relx=0.5, rely=0.5, anchor="center")  # Adjust relx value here

        # Canvas for displaying webcam feed
        self.canvas = tk.Canvas(self.recognition_frame, width=600, height=480, bg="white")
        self.canvas.pack()

        # Exit button
        self.exit_button = tk.Button(parent, text="Exit", command=self.close_window, font=("Times", 14), bg="red", fg="white")
        self.exit_button.place(relx=0.5, rely=0.9, anchor="center")

        self.update()

    def update(self):
        ret, img = self.cap.read()
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)

        if hands:
            # Initialize variables to store the coordinates of the combined bounding box
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            # Combine bounding boxes for all hands
            for hand in hands:
                x, y, w, h = hand['bbox']
                x_min = min(x_min, x - self.offset)
                y_min = min(y_min, y - self.offset)
                x_max = max(x_max, x + w + self.offset)
                y_max = max(y_max, y + h + self.offset)

            # Draw rectangle around the combined bounding box
            cv2.rectangle(imgOutput, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 4)

            # Crop the combined hand image
            combined_hand = img[int(y_min):int(y_max), int(x_min):int(x_max)]
            imgWhite = np.ones((self.imSize, self.imSize, 3), np.uint8) * 255

            # Resize the combined hand image to fit into the white image
            h, w, _ = combined_hand.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = self.imSize / h
                wCal = math.ceil(k * w)
                imgRsize = cv2.resize(combined_hand, (wCal, self.imSize))
                wgap = math.ceil((self.imSize - wCal) / 2)
                imgWhite[:, wgap:wCal + wgap] = imgRsize
            else:
                k = self.imSize / w
                hCal = math.ceil(k * h)
                imgRsize = cv2.resize(combined_hand, (self.imSize, hCal))
                hgap = math.ceil((self.imSize - hCal) / 2)
                imgWhite[hgap:hCal + hgap, :] = imgRsize

            # Classify the combined hand image
            prediction, index = self.classifier.getPrediction(imgWhite, draw=False)

            # Ensure index is within range before accessing labels list
            if 0 <= index < len(self.labels):
                label = self.labels[index]
            else:
                label = "Unknown"

            # Display the label above the hands
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.7, 2)[0]
            text_x = int((x_min + x_max - text_size[0]) / 2)
            text_y = int(y_min - 30)

            # Draw background rectangle for label text
            cv2.rectangle(imgOutput, (text_x - 10, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 10, text_y + text_size[1] + 5), (255, 0, 255), cv2.FILLED)

            # Draw the label text
            cv2.putText(imgOutput, label, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 2)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close_window()
        else:
            self.parent.after(10, self.update)

    def close_window(self):
        self.parent.destroy()


class MainWindow:
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("Sign Language Learning")

        # Get screen width and height
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()

        # Set window size and position
        self.parent.geometry(f"{screen_width}x{screen_height}+0+0")

        # Load and resize background image
        self.bg_image = Image.open("sign language ai image.png")
        self.bg_image = self.bg_image.resize((screen_width, screen_height))
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        # Create a canvas for the background image
        self.canvas = tk.Canvas(parent, width=screen_width, height=screen_height)
        self.canvas.pack()

        # Place the background image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_photo)

        # Create a custom font for welcome label
        welcome_font = font.Font(family="Times", size=40, weight="bold")

        # Buttons
        button_width = 29
        button_height = 3

        learn_button = tk.Button(parent, text="Start to Recognize and Learn", command=self.open_sign_recognition_window,
                                 font=("Times", 20, "bold"), bg="#FF5BAE", fg="white", width=button_width,
                                 height=button_height)
        learn_button.place(relx=0.5, rely=0.2, anchor="center")

    def open_sign_recognition_window(self):
        second_window = tk.Toplevel(self.parent)
        second_window.title("Sign Language Recognition")
        second_window.geometry(f"{self.bg_image.width}x{self.bg_image.height}+0+0")
        second_window.configure(background='white')

        SignRecognitionWindow(second_window)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
