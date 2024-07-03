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

with open(os.path.join(WEIGHTS_DIR, "classes.pkl"), "rb") as f:
    classes = pkl.load(f)

logger.info(f"Loaded classes -> {json.dumps(classes)}")

cap = cv2.VideoCapture(0)
mediapipeHelper = MediapipeHelper()
model = load_model_()

memory = np.zeros((1, FRAMES, FEATURES))

str_array = []
str_accumulated = ""

start_time = time.time()

t = 7

engine = pyttsx3.init()

def update_frame():
    global start_time, t, str_accumulated

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)

    elapsed_time = time.time() - start_time

    features_, frame = mediapipeHelper.getFeatures(frame)
    if len(features_):
        shift_top_inplace(memory, features_)

        preds = model.predict(memory)
        logger.info(f"Prediction Completed: {preds}")

        pred_index = np.argmax(preds)
        logger.info(f"Prediction Index: {pred_index}")

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

        prediction_text = f"Prediction: {classes[pred_index]} ({pred_confidence * 100:.2f}%)"
        prediction_label.config(text=prediction_text)
        str_label.config(text=f"Statement: {str_accumulated}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    video_label.after(10, update_frame)


def speak_statement():
    engine = pyttsx3.init()
    engine.setProperty('rate', 110)

    engine.say(str_accumulated)
    engine.runAndWait()

def clear_statement():
    global str_array, str_accumulated
    str_array.clear()
    str_accumulated = ""
    str_label.config(text=f"Statement: {str_accumulated}")

def train_model():
    from model.model import get_model, logger
    from data_loader import load_data
    import time

    start = time.time()
    X, y, n_sign = load_data()
    logger.info(f"Loading dataset took : {time.time() - start}")

    model = get_model(n_sign)

    logger.info(f"Training model on dataset --> {X.shape} and {y.shape}")
    history = model.fit(X, y, epochs=50)

    model.save("./weights/trained.keras", save_format='tf')
    logger.info("Model training completed and saved.")

def collect_data():
    sign = simpledialog.askstring("Input", "Enter sign name (or 'q' to quit):", parent=root)
    if sign == "q" or sign is None:
        return

    output_directory = 'Videos/'
    output_filename = sign + '.avi'
    output_path = os.path.join(output_directory, output_filename)
    os.makedirs(output_directory, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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

            if i < 5:
                out.write(frame)

            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                cap.release()
                return

        data_frames = np.array(data_frames)
        file_path = f"{DATASET_DIR}/{sign}_{i}.npy"
        np.save(file_path, data_frames)
        print(f"Dataset saved at --> {file_path} of shape --> {data_frames.shape}")
    cv2.destroyAllWindows()

def start_application():
    start_screen.destroy()

    global video_label, prediction_label, str_label, train_button, validate_button, collect_button, clear_button

    video_label = Label(root)
    video_label.pack()

    prediction_label = Label(root, text="Prediction: ", font=("Helvetica", 16))
    prediction_label.pack()

    str_label = Label(root, text="Statement: ", font=("Helvetica", 16))
    str_label.pack()

    train_button = Button(root, text="Train", command=train_model, width=10, height=4)
    train_button.pack()

    validate_button = Button(root, text="Validate", command=update_frame, width=10, height=4)
    validate_button.pack()

    collect_button = Button(root, text="Collect Data", command=collect_data, width=10, height=4)
    collect_button.pack()

    clear_button = Button(root, text="Clear", command=clear_statement, width=10, height=4)
    clear_button.pack()

    update_frame()

root = tk.Tk()
root.title("SIGN LANGUAGE TO AUDIO - GROUP 1")

start_screen = tk.Frame(root)
start_screen.pack(fill="both", expand=True)

title_label_1 = Label(start_screen, text="College of Technology, GBPUA&T, Pantnagar", font=("Helvetica", 27))
title_label_1.pack(pady=8)

title_label_3 = Label(start_screen, text="Department of Computer Engineering", font=("Helvetica", 24))
title_label_3.pack(pady=8)

title_label_2 = Label(start_screen, text="BTech Final Year Project", font=("Helvetica", 23))
title_label_2.pack(pady=8)

title_label = Label(start_screen, text="Group: 01", font=("Helvetica", 22))
title_label.pack(pady=8)

title_label_4 = Label(start_screen, text="Sign Language to Audio Conversion", font=("Helvetica", 22))
title_label_4.pack(pady=8)

photo_frame = tk.Frame(start_screen)
photo_frame.pack()

photos = ["photo1.jpg", "photo2.jpeg", "photo3.jpeg", "photo4.jpeg"]
photo_labels = ["Project Guide: Dr S.D. Samantaray", "Ayush Pratap Singh (56068)", "Ansh Kumar (56094)", "Shivanshu Rawat (56930)"]

for i, photo in enumerate(photos):
    img = Image.open(photo)
    img = img.resize((200, 200), Image.LANCZOS)
    photo_img = ImageTk.PhotoImage(img)

    img_label = Label(photo_frame, image=photo_img)
    img_label.image = photo_img
    img_label.grid(row=0, column=i, padx=10, pady=10)

    label_text = photo_labels[i]
    label = Label(photo_frame, text=label_text, font=("Helvetica", 18))
    label.grid(row=1, column=i, padx=10)


start_button = Button(start_screen, text="Start", command=start_application, width=10, height=2, font=("Helvetica", 18))
start_button.pack(pady=20)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
