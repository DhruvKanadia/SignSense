import streamlit as st
from PIL import Image
import os
import json
from streamlit_lottie import st_lottie

img_arg = Image.open("image/sign.png")
img_resize = img_arg.resize((1000, 1000))

def my_function():
    st.write("Button pressed!")

def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_coding = load_lottiefile("Animation/Ani.json")
img_arg = Image.open("image/sign.png")
img_resize = img_arg.resize((2000, 2000))

# HEADING
left_column, right_column = st.columns([3, 1])
with left_column:
    st.title("SignSense")
    st.write("Hand Gesture Recognition")
    st.write("")

with right_column:
    st.image(img_resize, use_column_width=True)
    
with st.container():
    st.write("---")
    left_column, right_column = st.columns([3, 1])
    with right_column:
        st.subheader("How you interact matters,")
        st.subheader("Do it together with SignSense.")
        st.subheader("")
        st.subheader("Language is just a medium for us to convey.")

    with left_column:
        st_lottie(
            lottie_coding,
            speed=0.25,
            loop=True,
            quality="low",
            height=300,
            width=500,
            key="coding",
        )

# Button
button_pressed = st.button("Extract!")

if button_pressed:
    st.write("Successfully Extracted Medical Terms")

# Additional content
st.write("---")
st.header("Our Vision")
st.write("Here you can describe your vision.")

st.write("---")
st.header("Business Proposal")
st.write("Here you can present your business proposal.")
