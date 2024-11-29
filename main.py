import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time

# Streamlit settings
st.set_page_config(page_title="Virtual Keyboard", layout="wide")
st.title("Interactive Virtual Keyboard")
st.subheader(
    """Turn on the webcam and use hand gestures to interact with the virtual keyboard.
    Use 'a' and 'd' from keyboard to change the background."""
)

# Virtual Keyboard Keys Layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

# Define Button Class
class Button:
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos
        self.size = size
        self.text = text

# Load Background Images
listImg = os.listdir('street')  # Ensure 'street' folder exists with background images
imgList = [cv2.imread(f'street/{imgPath}') for imgPath in listImg]
indexImg = 0  # Initial background index

# Define the Streamlit-WebRTC Transformer
class VirtualKeyboardTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.segmentor = SelfiSegmentation()
        self.prev_key_time = [time.time()] * 2
        self.output_text = ""
        self.indexImg = 0  # Background index
        self.buttonList = []
        
        # Define buttons for virtual keyboard
        for row_index, row in enumerate(keys):
            for col_index, key in enumerate(row):
                self.buttonList.append(Button([30 + col_index * 105, 30 + row_index * 120], key))
        self.buttonList.append(Button([1050, 30], 'BS', size=[125, 100]))
        self.buttonList.append(Button([300, 370], 'SPACE', size=[500, 100]))

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOut = self.segmentor.removeBG(img, imgList[self.indexImg], threshold=0.7)
        hands, img = self.detector.findHands(imgOut, flipType=False)
        keyboard_canvas = np.zeros_like(img)

        # Draw buttons on the virtual keyboard
        for button in self.buttonList:
            x, y = button.pos
            w, h = button.size
            cv2.rectangle(keyboard_canvas, button.pos, [x + w, y + h], (255, 0, 0), cv2.FILLED)
            cv2.putText(keyboard_canvas, button.text, (x + 20, y + 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

        # Hand detection and interaction with buttons
        if hands:
            for i, hand in enumerate(hands):
                lmList = hand["lmList"]
                if lmList:
                    x4, y4 = lmList[4][0], lmList[4][1]
                    x8, y8 = lmList[8][0], lmList[8][1]
                    distance = np.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2)

                    for button in self.buttonList:
                        x, y = button.pos
                        w, h = button.size

                        # Check if finger is on a button
                        if x < x8 < x + w and y < y8 < y + h:
                            cv2.rectangle(img, button.pos, [x + w, y + h], (0, 255, 0), -1)
                            cv2.putText(img, button.text, (x + 20, y + 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

                            # Handle button press when distance is within threshold
                            if distance < 40:
                                if time.time() - self.prev_key_time[i] > 2:
                                    self.prev_key_time[i] = time.time()
                                    if button.text == "BS":
                                        self.output_text = self.output_text[:-1]
                                    elif button.text == "SPACE":
                                        self.output_text += " "
                                    else:
                                        self.output_text += button.text

        # Return the final frame with keyboard overlay and output text
        cv2.putText(img, self.output_text, (30, 650), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        return img

# Streamlit WebRTC UI
webrtc_streamer(key="virtual-keyboard", video_transformer_factory=VirtualKeyboardTransformer)

# Background change logic for 'a' and 'd' keys
key = cv2.waitKey(1)
if key == ord('a'):
    if indexImg > 0:
        indexImg -= 1
elif key == ord('d'):
    if indexImg < len(imgList) - 1:
        indexImg += 1
