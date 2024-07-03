import cv2
import numpy as np
import os
import pickle as pkl
import time
import tkinter as tk
import json
import pyttsx3
from tkinter import Label, Button, simpledialog
from PIL import Image, ImageTk

from utils.mediapipe_helper import MediapipeHelper
from model.model import load_model_, get_model, logger
from data_loader import load_data
from get_logger import *
from utils.helper import shift_top_inplace

DATASET_DIR = "./dataset"
FONT_SIZE = 1
FRAMES = int(1.5 * 24)
FEATURES = 84
SAMPLES = 25
WEIGHTS_DIR = "./weights"

indian_states = [
    "Andhra Pradesh",
    "India",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal"
]

indian_state_capitals = [
    "Amaravati",
    "New Delhi",
    "Itanagar",
    "Dispur",
    "Patna",
    "Raipur",
    "Panaji",
    "Gandhinagar",
    "Chandigarh",
    "Shimla",
    "Ranchi",
    "Bengaluru",
    "Thiruvananthapuram",
    "Bhopal",
    "Mumbai",
    "Imphal",
    "Shillong",
    "Aizawl",
    "Kohima",
    "Bhubaneswar",
    "Chandigarh",
    "Jaipur",
    "Gangtok",
    "Chennai",
    "Hyderabad",
    "Agartala",
    "Lucknow",
    "Dehradun",
    "Kolkata"
]

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

str_array = []
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
            if classes[pred_index] == "OK":
                speak_statement()
                str_accumulated = ""
                str_array.clear()
            else:
                str_array.append(classes[pred_index])

                if str_array[-1] == "capital":
                    str_array.append("of")
                elif str_array[-1] in indian_state_capitals:
                    str_array.append("is the")

                str_accumulated = " ".join(str_array)

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
    engine = pyttsx3.init()
    # Set the rate of speech to a definite value (e.g., 150 words per minute)
    engine.setProperty('rate', 110)

    engine.say(str_accumulated)
    engine.runAndWait()

# Function to clear the accumulated string and array
def clear_statement():  # Highlighted
    global str_array, str_accumulated
    str_array.clear()
    str_accumulated = ""
    str_label.config(text=f"Statement: {str_accumulated}")

# Function to train the model
def train_model():
    from model.model import get_model, logger
    from data_loader import load_data
    import time

    start = time.time()
    # Read data from /dataset and send out X->(BATCH, FRAMES, FEATURES), y->labels
    X, y, n_sign = load_data()
    logger.info(f"Loading dataset took : {time.time() - start}")

    model = get_model(n_sign)

    logger.info(f"Training model on dataset --> {X.shape} and {y.shape}")
    history = model.fit(X, y, epochs=50)

    model.save("./weights/trained.keras", save_format='tf')
    logger.info("Model training completed and saved.")

# Function to collect data
def collect_data():
    sign = simpledialog.askstring("Input", "Enter sign name (or 'q' to quit):", parent=root)
    if sign == "q" or sign is None:
        return

    # Specify the directory where you want to save the video
    output_directory = 'Videos/'  # Replace with your desired folder path
    output_filename = sign + '.avi'
    output_path = os.path.join(output_directory, output_filename)
    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs as well (e.g., 'MJPG', 'X264', etc.)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    for i in range(SAMPLES):
        data_frames = []
        ctr = 0
        while ctr < FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            features, frame = mediapipeHelper.getFeatures(frame)
            if len(features) != 0:
                ctr += 1
                data_frames.append(features)

            cv2.putText(frame, f"HANDS: {ctr}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255))
            cv2.putText(frame, f"ROUND: {i+1}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255))
            cv2.imshow("Data Collection", frame)

            # Write the frame into the file
            if i < 5:
                out.write(frame)

            if cv2.waitKey(1) == 27:  # Press 'ESC' to quit
                cv2.destroyAllWindows()
                cap.release()
                return

        data_frames = np.array(data_frames)
        file_path = f"{DATASET_DIR}/{sign}_{i}.npy"
        np.save(file_path, data_frames)
        print(f"Dataset saved at --> {file_path} of shape --> {data_frames.shape}")
    cv2.destroyAllWindows()

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

# Train button
train_button = Button(root, text="Train", command=train_model, width=10, height=4)
train_button.pack()

# Validate button
validate_button = Button(root, text="Validate", command=update_frame, width=10, height=4)
validate_button.pack()

# Data Collection button
collect_button = Button(root, text="Collect Data", command=collect_data, width=10, height=4)
collect_button.pack()

# Clear button
clear_button = Button(root, text="Clear", command=clear_statement, width=10, height=4)  # Highlighted
clear_button.pack()

# Start the Tkinter main loop
root.mainloop()

# Release the video capture when the Tkinter window is closed
cap.release()
cv2.destroyAllWindows()
