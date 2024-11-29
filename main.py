import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from HandTrackingModule import HandDetector  # Make sure to import your HandDetector from a separate module if needed
import os

# Set up virtual keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

# Define Button class for virtual keyboard
class Button:
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos  # Position of button
        self.size = size  # Size of button
        self.text = text  # Text displayed on button

    def draw(self, img):
        cv2.putText(img, self.text, (self.pos[0] + 20, self.pos[1] + 70), 
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

# Load background images for segmentation (optional feature)
listImg = os.listdir('street')
imgList = [cv2.imread(f'street/{imgPath}') for imgPath in listImg]

# Initialize webcam and HandDetector
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
detector = HandDetector(maxHands=2, detectionCon=0.8)

finalText = ''  # Store the final text typed

# VideoProcessor class for streamlit_webrtc
class VideoProcessor:
    def __init__(self):
        self.indexImg = 0  # Background index
        self.segmentor = SelfiSegmentation()  # Background segmentation model
        self.prev_key_time = [time.time()] * 2  # Track key press times

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Remove background from the frame
        imgOut = self.segmentor.removeBG(frm, imgList[self.indexImg])

        # Detect hands in the frame
        hands, img = detector.findHands(imgOut)

        keyboard_canvas = np.zeros_like(img)
        buttonList = []

        # Draw virtual keyboard
        for key in keys[0]:
            buttonList.append(Button([30 + keys[0].index(key) * 105, 30], key))
        for key in keys[1]:
            buttonList.append(Button([30 + keys[1].index(key) * 105, 150], key))
        for key in keys[2]:
            buttonList.append(Button([30 + keys[2].index(key) * 105, 260], key))

        # Special buttons (Backspace, Space)
        buttonList.append(Button([90 + 10 * 100, 30], 'BS', size=[125, 100]))
        buttonList.append(Button([300, 370], 'SPACE', size=[500, 100]))

        # Process each detected hand
        for i, hand in enumerate(hands):
            lmList = hand["lmList"]  # Get list of landmarks for the hand
            if lmList:
                x4, y4 = lmList[4][0], lmList[4][1]  # Coordinates of the thumb
                x8, y8 = lmList[8][0], lmList[8][1]  # Coordinates of the index finger

                # Calculate distance between thumb and index finger
                distance = np.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2)
                click_threshold = 10  # Distance threshold for click detection

                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    # Check if fingertip is over a button
                    if x < x8 < x + w and y < y8 < y + h:
                        cv2.rectangle(img, button.pos,
                                      [button.pos[0] + button.size[0], button.pos[1] + button.size[1]],
                                      (0, 255, 160), -1)
                        button.draw(img)

                        if distance < click_threshold:
                            if time.time() - self.prev_key_time[i] > 3.5:  # Prevent repeated key presses
                                self.prev_key_time[i] = time.time()
                                # Update text based on button pressed
                                if button.text != 'BS' and button.text != 'SPACE':
                                    finalText += button.text
                                elif button.text == 'BS':
                                    finalText = finalText[:-1]
                                else:
                                    finalText += ' '

        # Show the final typed text
        cv2.putText(img, finalText, (120, 580), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        return av.VideoFrame.from_ndarray(img, format='bgr24')

# Initialize Streamlit WebRTC streamer
webrtc_streamer(key="key", 
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

