import cv2
import numpy as np
import os
import pickle as pkl
import time
import tkinter as tk
import json
import pyttsx3
from tkinter import Label, Button
from PIL import Image, ImageTk

from utils.mediapipe_helper import MediapipeHelper
from constants import *
from model.model import load_model_
from get_logger import *
from utils.helper import shift_top_inplace

# Load the classes from a pickle file
with open(os.path.join(WEIGHTS_DIR, "classes.pkl"), "rb") as f:
    classes = pkl.load(f)

logger.info(f"Loaded classes -> {json.dumps(classes)}")

# Initialize video capture
cap = cv2.VideoCapture(0)
mediapipeHelper = MediapipeHelper()
model = load_model_()

# Initialize memory for storing features
memory = np.zeros((1, FRAMES, FEATURES))

str_accumulated = ""

# Initialize start time
start_time = time.time()

t = 7

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Function to update the frame
def update_frame():
    global start_time, t, str_accumulated

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Get features from the frame
    features_, frame = mediapipeHelper.getFeatures(frame)
    if len(features_):
        # Update memory with the new features
        shift_top_inplace(memory, features_)

        # Make predictions
        preds = model.predict(memory)
        logger.info(f"Prediction Completed: {preds}")

        # Get the index of the highest prediction
        pred_index = np.argmax(preds)
        logger.info(f"Prediction Index: {pred_index}")

        # Get the confidence score of the predicted label
        pred_confidence = preds[0][pred_index]

        if elapsed_time > t and (pred_confidence * 100) > 90:
            t = elapsed_time + 10
            if classes[pred_index] != "Space":
                str_accumulated += classes[pred_index]
            else:
                str_accumulated += " "

        # Update the prediction text and accumulated string
        prediction_text = f"Prediction: {classes[pred_index]} ({pred_confidence * 100:.2f}%)"
        prediction_label.config(text=prediction_text)
        str_label.config(text=f"Statement: {str_accumulated}")

    # Convert the frame to ImageTk format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    # Schedule the next frame update
    video_label.after(10, update_frame)

# Function to speak the accumulated string
def speak_statement():
    engine.say(str_accumulated)
    engine.runAndWait()

#def clear_statement():
    #str_accumulated = ""

#def backspace():
    #str_accumulated =  str_accumulated[:-1]

# Setup the Tkinter window
root = tk.Tk()
root.title("SIGN LANGUAGE TO AUDIO - GROUP 1")

# Video display label
video_label = Label(root)
video_label.pack()

# Prediction text label
prediction_label = Label(root, text="Prediction: ", font=("Helvetica", 16))
prediction_label.pack()

# Accumulated string label
str_label = Label(root, text="Statement: ", font=("Helvetica", 16))
str_label.pack()

# Backspace button
#backspace_button = Button(root, text="Backspace", command=backspace, width=10, height=4)
#backspace_button.pack()

# Speak button
#clear_button = Button(root, text="Clear", command=clear_statement, width=10, height=4)
#clear_button.pack()

# Speak button
speak_button = Button(root, text="Speak", command=speak_statement, width=10, height=4)
speak_button.pack()

# Start updating the frame
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Release the video capture when the Tkinter window is closed
cap.release()
cv2.destroyAllWindows()
