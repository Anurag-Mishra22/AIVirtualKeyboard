import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time

# Streamlit page configuration
st.set_page_config(page_title="Virtual Keyboard", layout="wide")
st.title("Interactive Virtual Keyboard")
st.subheader(
    "Turn on the webcam and use hand gestures to interact with the virtual keyboard."
    "\nUse 'a' and 'd' keys to change the background."
)

# Load background images from the 'street' directory
background_dir = 'street'
if not os.path.exists(background_dir):
    st.error(f"Error: The directory '{background_dir}' is missing. Please add background images.")
    st.stop()

background_images = []
for img_file in os.listdir(background_dir):
    img_path = os.path.join(background_dir, img_file)
    img = cv2.imread(img_path)
    if img is not None:
        background_images.append(img)
    else:
        st.error(f"Error: Could not load image {img_file}.")

if not background_images:
    st.error(f"Error: No valid images found in the '{background_dir}' directory.")
    st.stop()

# Hand detector and segmentation module
hand_detector = HandDetector(maxHands=1, detectionCon=0.8)
segmentor = SelfiSegmentation()

# Define virtual keyboard layout
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
]

class Button:
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos
        self.size = size
        self.text = text

# Streamlit session-specific variables
indexImg = 0
output_text = ""
prev_key_time = time.time()

def video_frame_callback(frame):
    global indexImg, output_text, prev_key_time

    # Convert the input video frame to an OpenCV image
    img = frame.to_ndarray(format="bgr24")
    img_out = segmentor.removeBG(img, background_images[indexImg])

    # Detect hands
    hands, img_out = hand_detector.findHands(img_out, flipType=False)

    # Draw virtual keyboard
    keyboard_canvas = np.zeros_like(img_out)
    button_list = []

    for key in keys[0]:
        button_list.append(Button([30 + keys[0].index(key) * 105, 30], key))
    for key in keys[1]:
        button_list.append(Button([30 + keys[1].index(key) * 105, 150], key))
    for key in keys[2]:
        button_list.append(Button([30 + keys[2].index(key) * 105, 260], key))

    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(keyboard_canvas, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(
            keyboard_canvas,
            button.text,
            (x + 20, y + 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            2,
        )

    # Interact with the keyboard based on detected hand landmarks
    if hands:
        lm_list = hands[0]["lmList"]
        if lm_list:
            x8, y8 = lm_list[8][0], lm_list[8][1]  # Index finger tip
            for button in button_list:
                x, y = button.pos
                w, h = button.size
                if x < x8 < x + w and y < y8 < y + h:
                    cv2.rectangle(keyboard_canvas, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(
                        keyboard_canvas,
                        button.text,
                        (x + 20, y + 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 255, 255),
                        2,
                    )
                    # Click detection
                    distance = np.linalg.norm(
                        np.array(lm_list[8][:2]) - np.array(lm_list[4][:2])
                    )
                    if distance < 30 and time.time() - prev_key_time > 1:
                        prev_key_time = time.time()
                        if button.text != "BS" and button.text != "SPACE":
                            output_text += button.text
                        elif button.text == "BS":
                            output_text = output_text[:-1]
                        elif button.text == "SPACE":
                            output_text += " "

    # Combine the original image with the keyboard overlay
    stacked_img = cv2.addWeighted(img_out, 0.7, keyboard_canvas, 0.3, 0)
    return av.VideoFrame.from_ndarray(stacked_img, format="bgr24")

# Start the webcam stream with webrtc_streamer
webrtc_streamer(
    key="virtual_keyboard",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)

# Display the output text in the Streamlit app
st.subheader("Output Text")
st.write(output_text)
