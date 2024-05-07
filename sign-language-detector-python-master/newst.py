import streamlit as st
from PIL import Image
import os
import json
from streamlit_lottie import st_lottie
import pickle

import cv2
import mediapipe as mp
import numpy as np


cap = cv2.VideoCapture(0)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 
               23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 
               34: '8', 35: '9', 36: 'I Love You', 37: 'EAT', 38: 'OK', 39: 'HELP'}

img_arg = Image.open("image/sign.png")
img_resize=img_arg.resize((1000,1000))

def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)


lottie_coding = load_lottiefile("Animation/Ani.json")
img_arg = Image.open("image/sign.png")
img_resize=img_arg.resize((2000,2000))

#HEADING
left_column, right_column,west_column = st.columns([1,2,3])
with left_column:
    st.image(img_resize,use_column_width=True)

with right_column:
    st.title("SignSense")
    st.write("Hand Gesture Recognition") 
    st.write("") 
with west_column:
        st_lottie(
            lottie_coding,
            speed=0.25,
            loop=True,
            quality="low",
            height=300,
            width=500,
            key="coding",
        )

with st.container():
    st.write("---")
    st.subheader("How you interact matters,")
    st.subheader("Do it together with SignSense.")
    
with st.container():
    st.write("---")
    left_column, right_column = st.columns([1,1])
    with left_column:
        button_pressed = st.button("Open Camera!")  
        st.write("")
    with right_column:
        button_pressed1 = st.button("Close Camera!")
        
        
    





if button_pressed:
    st.write("Successfully Opened  Camera")
    camera_box = st.empty()
    
    while True:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_box.image(frame, channels="RGB")
        if button_pressed1:
            break

    cap.release()
    cv2.destroyAllWindows()

if button_pressed1:
    cap.release()
    cv2.destroyAllWindows()
