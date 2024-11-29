import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
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
    "Turn on the webcam and use hand gestures to interact with the virtual keyboard.\n"
    "Use 'a' and 'd' to change the background."
)

# Initialize Hand Detector and Segmentor
detector = HandDetector(maxHands=1, detectionCon=0.8)
segmentor = SelfiSegmentation()

# Load background images
bg_dir = "street"
if os.path.exists(bg_dir):
    bg_images = [cv2.imread(os.path.join(bg_dir, img)) for img in os.listdir(bg_dir) if img.endswith(('.jpg', '.png'))]
else:
    bg_images = []

if not bg_images:
    st.error("No background images found. Please add images to the 'street' directory.")
    st.stop()

indexImg = 0  # Default background index
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
]


# Virtual keyboard button class
class Button:
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos
        self.size = size
        self.text = text


# Create buttons for the keyboard
buttonList = []
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        buttonList.append(Button([30 + j * 105, 30 + i * 120], key))

buttonList.append(Button([1100, 30], "BS", [125, 100]))  # Backspace button
buttonList.append(Button([300, 370], "SPACE", [500, 100]))  # Space button

# WebRTC Video Transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.output_text = ""
        self.prev_key_time = time.time()

    def transform(self, frame):
        global indexImg

        # Read and preprocess the video frame
        img = frame.to_ndarray(format="bgr24")
        imgOut = segmentor.removeBG(img, bg_images[indexImg], threshold=0.8)

        # Detect hands
        hands, img = detector.findHands(imgOut, flipType=False)

        # Create a blank canvas for the keyboard
        keyboard_canvas = np.zeros_like(img)

        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            cv2.rectangle(keyboard_canvas, button.pos, (x + w, y + h), (50, 50, 50), -1)
            cv2.putText(keyboard_canvas, button.text, (x + 20, y + 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

        # Process hand gestures
        if hands:
            for hand in hands:
                lmList = hand["lmList"]
                x8, y8 = lmList[8][:2]  # Index fingertip
                x4, y4 = lmList[4][:2]  # Thumb tip
                distance = np.linalg.norm([x8 - x4, y8 - y4])

                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size
                    if x < x8 < x + w and y < y8 < y + h:
                        cv2.rectangle(keyboard_canvas, button.pos, (x + w, y + h), (0, 255, 0), -1)
                        cv2.putText(keyboard_canvas, button.text, (x + 20, y + 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

                        # Simulate a click
                        if distance < 40 and time.time() - self.prev_key_time > 1.5:
                            self.prev_key_time = time.time()
                            if button.text == "BS":
                                self.output_text = self.output_text[:-1]
                            elif button.text == "SPACE":
                                self.output_text += " "
                            else:
                                self.output_text += button.text

        # Overlay the keyboard on the frame
        img = cv2.addWeighted(img, 0.7, keyboard_canvas, 0.3, 0)

        # Display output text
        cv2.putText(img, self.output_text, (50, 600), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        return img


# WebRTC Streamlit Integration
webrtc_streamer(
    key="virtual-keyboard",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=VideoTransformer
)

# Sidebar controls for changing background
st.sidebar.title("Controls")
if st.sidebar.button("Previous Background"):
    indexImg = max(0, indexImg - 1)
if st.sidebar.button("Next Background"):
    indexImg = min(len(bg_images) - 1, indexImg + 1)
