# Imports
import time
from tkinter import *
import cv2
from PIL import Image, ImageTk
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import hand_tracking_module as htm
import os
import numpy as np


class App:
    def __init__(self, video_source=0):
        self.player = QMediaPlayer()  # Initialize player for click sound
        self.app_name = "Virtual Paint"
        self.window = Tk()
        self.window.title(self.app_name)
        self.window.resizable(0, 0)
        self.window.wm_iconbitmap("paint.ico")
        self.window["bg"] = "#FBC088"
        self.video_source = video_source
        self.photo = None

        self.vid = MyVideoCapture(self.video_source)
        self.label = Label(self.window, text=self.app_name, font=20, bg="gray", fg="white").pack(side=TOP, fill=BOTH)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(self.window, width=1280, height=720)
        self.canvas.pack()

        # Button which takes a snapshot
        self.btn_snapshot = Button(self.window, text="Save", font="lucida 15 bold", width=30, bg="#8FB48E",
                                   activebackground="#564BF7",
                                   command=self.snapshot)
        self.btn_snapshot.pack(anchor=CENTER, expand=True)
        self.update()
        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        check, frame = self.vid.getFrame()
        if check:
            image = "IMG-" + time.strftime("%H-%M-%S-%d-%m") + ".jpg"
            cv2.imwrite(image, cv2.cvtColor(frame[125:, 0:1280], cv2.COLOR_BGR2RGB))

            # Show the message on window that image was saved
            # msg = Label(self.window, text="Image Saved: " + image, font="lucida 10 bold", bg="#EBC16A", fg="#36454F")
            # msg.place(x=710, y=900)
            # Sound
            file = QUrl("click.wav")
            content = QMediaContent(file)
            self.player.setMedia(content)
            self.player.play()

    def update(self):
        # Get a frame from the video source
        isTrue, frame = self.vid.getFrame()
        if isTrue:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        self.window.after(15, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open Camera")

        # Get video source width and height
        # self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.brushThickness = 25
        self.eraserThickness = 100

        self.folderPath = "img"
        self.myList = os.listdir(self.folderPath)
        self.overlayList = []
        for imPath in self.myList:
            self.image = cv2.imread(f'{self.folderPath}/{imPath}')
            self.overlayList.append(self.image)
        # print(len(self.overlayList))
        self.header = self.overlayList[0]
        self.drawColor = (255, 0, 255)

        self.vid.set(3, 1280)
        self.vid.set(4, 720)

        self.detector = htm.handDetector(detectionCon=0.50, maxHands=1)
        self.xp, self.yp = 0, 0
        self.imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    def getFrame(self):
        if self.vid.isOpened():
            # 1. Import image
            success, img = self.vid.read()
            img = cv2.flip(img, 1)

            # 2. Find Hand Landmarks
            img = self.detector.findHands(img)
            lmList = self.detector.findPosition(img, draw=False)

            if len(lmList) != 0:

                # print(lmList)

                # tip of index and middle fingers
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]

                # 3. Check which fingers are up
                fingers = self.detector.fingers_up()
                # print(fingers)

                # 4. If Selection Mode - Two finger are up
                if fingers[1] and fingers[2]:
                    self.xp, self.yp = 0, 0
                    print("Selection Mode")
                    # # Checking for the click
                    if y1 < 125:
                        if 250 < x1 < 450:
                            self.header = self.overlayList[0]
                            self.drawColor = (255, 0, 255)
                        elif 550 < x1 < 750:
                            self.header = self.overlayList[1]
                            self.drawColor = (255, 0, 0)
                        elif 800 < x1 < 950:
                            self.header = self.overlayList[2]
                            self.drawColor = (0, 255, 0)
                        elif 1050 < x1 < 1200:
                            self.header = self.overlayList[3]
                            self.drawColor = (0, 0, 0)
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)

                    # 5. If Drawing Mode - Index finger is up
                if fingers[1] and fingers[2] == False:
                    cv2.circle(img, (x1, y1), 15, self.drawColor, cv2.FILLED)
                    print("Drawing Mode")
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x1, y1

                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)

                    if self.drawColor == (0, 0, 0):
                        cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
                        cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)

                    else:
                        cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)
                        cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)

                    self.xp, self.yp = x1, y1

                    # Clear Canvas when all fingers are up
                # if all (x >= 1 for x in fingers):
                #     self.imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, self.imgCanvas)

            # Setting the header image
            img[0:125, 0:1280] = self.header

            if success:
                # If isTrue is true then current frame converted to RGB
                return success, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                return success, None

    def __del__(self):
        self.vid.release()


if __name__ == "__main__":
    App(0)
