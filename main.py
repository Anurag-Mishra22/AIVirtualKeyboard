import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time

# Streamlit settings
st.set_page_config(page_title="Virtual Keyboard", layout="wide")
st.title("Interactive Virtual Keyboard")
st.subheader('''Turn on the webcam and use hand gestures to interact with the virtual keyboard.
Use 'a' and 'd' from the keyboard to change the background.
''')

# Define virtual keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

# Class to define button properties for virtual keyboard
class Button:
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos  # Position of button
        self.size = size  # Size of button
        self.text = text  # Text displayed on button

    def draw(self, img):
        # Draw button text on the keyboard canvas
        cv2.putText(img, self.text, (self.pos[0] + 20, self.pos[1] + 70),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

# Video processing class for Streamlit WebRTC
class VideoProcessor:
    def __init__(self):
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.segmentor = SelfiSegmentation()
        self.imgList = self.load_background_images()
        self.indexImg = 0
        self.prev_key_time = [time.time()] * 2  # Time tracker for key press delay
        self.output_text = ""

    def load_background_images(self):
        listImg = os.listdir('street') if os.path.exists('street') else []
        imgList = []
        if not listImg:
            st.error("Error: 'street' directory is missing or empty. Please add background images.")
            st.stop()
        for imgPath in listImg:
            image = cv2.imread(f'street/{imgPath}')
            if image is None:
                st.error(f"Error: Failed to load image: {imgPath}")
            else:
                imgList.append(image)
        return imgList

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Apply segmentation and hand detection
        imgOut = self.segmentor.removeBG(frm, self.imgList[self.indexImg])  # Default in BGR format
        hands, img = self.detector.findHands(imgOut, flipType=False)

        # Create a blank canvas for drawing the keyboard
        keyboard_canvas = np.zeros_like(img)
        buttonList = []

        # Define buttons in each row of the virtual keyboard
        for key in keys[0]:
            buttonList.append(Button([30 + keys[0].index(key) * 105, 30], key))
        for key in keys[1]:
            buttonList.append(Button([30 + keys[1].index(key) * 105, 150], key))
        for key in keys[2]:
            buttonList.append(Button([30 + keys[2].index(key) * 105, 260], key))

        # Add special buttons for Backspace and Space
        buttonList.append(Button([90 + 10 * 100, 30], 'BS', size=[125, 100]))
        buttonList.append(Button([300, 370], 'SPACE', size=[500, 100]))

        # Use white color (255, 255, 255) for special buttons
        cv2.putText(keyboard_canvas, 'BS', (90 + 10 * 100 + 20, 30 + 70),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)
        cv2.putText(keyboard_canvas, 'SPACE', (300 + 20, 370 + 70),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

        # Check if hands are detected
        for i, hand in enumerate(hands):
            bbox = hand['bbox']  # Get bounding box for hand
            hand_width, hand_height = bbox[2], bbox[3]  # Extract hand dimensions
            lmList = hand["lmList"]  # Get list of landmarks for the hand

            if lmList:
                # Calculate the distance between landmarks 6 and 8
                x4, y4 = lmList[4][0], lmList[4][1]
                x8, y8 = lmList[8][0], lmList[8][1]
                distance = np.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2)
                click_threshold = 10  # Adjust the threshold for click detection

                # Loop through buttons and check if fingertip is over a button
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    # Check if index finger tip is within the button bounds
                    if x < x8 < x + w and y < y8 < y + h:
                        # Highlight the button being pointed at by index finger
                        cv2.rectangle(img, button.pos,
                                      [button.pos[0] + button.size[0], button.pos[1] + button.size[1]],
                                      (0, 255, 160), -1)
                        button.draw(img)

                        # Check for click gesture
                        if (distance / np.sqrt((hand_width) ** 2 + (hand_height) ** 2)) * 100 < click_threshold:
                            if time.time() - self.prev_key_time[i] > 2:  # Time delay for key press
                                self.prev_key_time[i] = time.time()  # Update key press time
                                cv2.rectangle(img, button.pos,
                                              [button.pos[0] + button.size[0], button.pos[1] + button.size[1]],
                                              (9, 9, 179), -1)
                                button.draw(img)

                                # Update output text based on button pressed
                                if button.text != 'BS' and button.text != 'SPACE':
                                    self.output_text += button.text
                                elif button.text == 'BS':
                                    self.output_text = self.output_text[:-1]
                                else:
                                    self.output_text += ' '

        # Overlay keyboard on webcam frame
        stacked_img = cv2.addWeighted(img, 0.7, keyboard_canvas, 0.3, 0)
        return av.VideoFrame.from_ndarray(stacked_img, format='bgr24')

# Streamlit WebRTC streamer configuration
webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ))

