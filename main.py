import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time
import os

# Streamlit settings
st.set_page_config(page_title="Virtual Keyboard", layout="wide")
st.title("Interactive Virtual Keyboard")
st.subheader('''Turn on the webcam and use hand gestures to interact with the virtual keyboard.
Use 'a' and 'd' from keyboard to change the background.
        ''')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Webcam not detected. Please check the connection or refresh and try again.")
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)
segmentor = SelfiSegmentation()

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
        # Draw button text on the keyboard canvas
        cv2.putText(img, self.text, (self.pos[0]+20, self.pos[1]+70),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

listImg=os.listdir('street')
imgList = []
for imgPath in listImg:
    image = cv2.imread(f'street/{imgPath}')
    imgList.append(image)

indexImg=0
segmentor = SelfiSegmentation()  # Initialize segmentation model
prev_key_time = [time.time()] * 2  # Initialize time tracker for key press delay for both hands

pressed_key=None
output_text=""

# Streamlit layout
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox("Run Webcam", value=True)
    FRAME_WINDOW = st.image([])
with col2:
    st.subheader("Output Text")
    output_text_area = st.subheader("")

while run:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg])  # it is default in BGR format
    if not success:
        st.error("Failed to read from webcam.")
        break
    # Detect hands
    hands, img = detector.findHands(imgOut, flipType=False)
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
            cv2.putText(img, f"Distance: {distance}", (50, 700), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv2.putText(img, f"Ratio: {(distance/np.sqrt((hand_width) ** 2 + (hand_height) ** 2))*100}", (50, 650), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            click_threshold = 10 # Adjust the threshold for click detection

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
                    cv2.putText(img, button.text, (button.pos[0] + 20, button.pos[1] + 70),
                                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)
                    # Check for click gesture
                    if (distance/np.sqrt((hand_width) ** 2 + (hand_height) ** 2))*100 < click_threshold:
                        if time.time() - prev_key_time[i] > 2 :  # Check time delay for key press previously 3.5
                            prev_key_time[i] = time.time()  # Update the last key press time for that hand
                            cv2.rectangle(img, button.pos,
                                          [button.pos[0] + button.size[0], button.pos[1] + button.size[1]],
                                          (9, 9, 179), -1)
                            cv2.putText(img, button.text, (button.pos[0] + 20, button.pos[1] + 70),
                                        cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

                            # Update final text based on button pressed
                            if button.text != 'BS' and button.text != 'SPACE':
                                output_text += button.text
                            elif button.text == 'BS':
                                output_text = output_text[:-1]
                            else:
                                output_text += ' '
    FRAME_WINDOW.image(img, channels="BGR")
    if output_text:
        output_text_area.text(output_text)


    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    key=cv2.waitKey(1)

    if key == ord('a'):
        if indexImg > 0:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg < len(imgList) - 1:
            indexImg += 1
    elif key == ord('q'):
        break

    stacked_img = cv2.addWeighted(img, 0.7, keyboard_canvas, 0.3, 0)






