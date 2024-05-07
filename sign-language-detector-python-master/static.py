import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os

# Define the path to the model file
model_path = r"C:\Users\HP\Downloads\my_model.keras"

# Verify file existence and accessibility
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Load your trained Keras model
try:
    cnn = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Mapping of class indices to alphabet letters
alphabet_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Function to make prediction
def predict_letter(image_path, model):
    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Make prediction
    result = model.predict(test_image)
    predicted_class_index = np.argmax(result)

    # Map class index to alphabet letter
    predicted_letter = alphabet_mapping.get(predicted_class_index, 'Unknown')
    
    return predicted_letter

# Streamlit app
def main():
    st.title("ASL Alphabet Image Classifier")

    # List to store predicted alphabet letters
    predicted_letters = []

    # File uploader for multiple image inputs
    uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write("Number of uploaded files:", len(uploaded_files))  # Debug statement

        for uploaded_file in uploaded_files:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Save the uploaded image temporarily
            image_path = os.path.join('./temp', uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Make prediction for the uploaded image
            try:
                predicted_letter = predict_letter(image_path, cnn)
                predicted_letters.append(predicted_letter)
                st.write(f"Predicted letter: {predicted_letter}")  # Debug statement
            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Display the composed sentence or word
        if predicted_letters:
            composed_text = ''.join(predicted_letters)
            st.success(f"Predicted word: {composed_text}")

if __name__ == "__main__":
    main()