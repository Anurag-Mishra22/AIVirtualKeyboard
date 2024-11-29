import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time

# Streamlit settings
st.set_page_config(page_title="Virtual Keyboard", layout="wide")
st.title("Interactive Virtual Keyboard")

# Initialize hand detector and selfi segmentation
detector = HandDetector(maxHands=1, detectionCon=0.8)
segmentor = SelfiSegmentation()

# Define virtual keyboard
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

# Load background images
listImg = os.listdir('street')
imgList = [cv2.imread(f'street/{imgPath}') for imgPath in listImg]
indexImg = 0

# Shared state for output text
if "output_text" not in st.session_state:
    st.session_state["output_text"] = ""

# Process video frame callback function
def process_video_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    imgOut = segmentor.removeBG(img, imgList[indexImg])  # Remove background

    # Detect hands in the frame
    hands, img = detector.findHands(imgOut)

    # Create virtual keyboard
    keyboard_canvas = np.zeros_like(img)
    buttonList = []

    # Draw buttons
    for key in keys[0]:
        buttonList.append(Button([30 + keys[0].index(key) * 105, 30], key))
    for key in keys[1]:
        buttonList.append(Button([30 + keys[1].index(key) * 105, 150], key))
    for key in keys[2]:
        buttonList.append(Button([30 + keys[2].index(key) * 105, 260], key))

    # Handle hand gestures and key press
    for hand in hands:
        bbox = hand['bbox']
        hand_width, hand_height = bbox[2], bbox[3]
        lmList = hand["lmList"]

        if lmList:
            x4, y4 = lmList[4][0], lmList[4][1]
            x8, y8 = lmList[8][0], lmList[8][1]
            distance = np.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2)

            click_threshold = 10  # Adjust threshold for click detection

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < x8 < x + w and y < y8 < y + h:
                    # Button press logic
                    if (distance/np.sqrt((hand_width) ** 2 + (hand_height) ** 2))*100 < click_threshold:
                        if time.time() - prev_key_time[i] > 3.5:  # Prevent key spam
                            prev_key_time[i] = time.time()  # Update the last key press time
                            # Update final text logic based on button press
                            if button.text != 'BS' and button.text != 'SPACE':
                                st.session_state["output_text"] += button.text
                            elif button.text == 'BS':
                                st.session_state["output_text"] = st.session_state["output_text"][:-1]
                            else:
                                st.session_state["output_text"] += ' '

    # Display the typed text on screen
    cv2.putText(img, st.session_state["output_text"], (120, 580), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    # Combine background video with keyboard display
    stacked_img = cv2.addWeighted(img, 0.7, keyboard_canvas, 0.3, 0)

    return stacked_img

# WebRTC streamer for video capture
webrtc_streamer(
    key="virtual-keyboard",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=process_video_frame,
)

