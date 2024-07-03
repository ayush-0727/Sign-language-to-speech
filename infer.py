import cv2
from utils.mediapipe_helper import MediapipeHelper
from constants import *
import numpy as np
from model.model import load_model_
from get_logger import *
from utils.helper import shift_top_inplace
import pickle as pkl
import os
import json
import time

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

str = ""

# Initialize start time
start_time = time.time()

t = 5

while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Display the elapsed time on the frame
    time_text = f"Time: {elapsed_time:.2f} sec"
    cv2.putText(frame, time_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), 2)

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

        if(elapsed_time > t and ((pred_confidence * 100)>90)):
            t = elapsed_time + 7
            if(classes[pred_index] != "Space"):
                str = str + classes[pred_index]
            else:
                str = str + " "

        # Display the predicted class and confidence score on the frame
        prediction_text = f"Prediction: {classes[pred_index]} ({pred_confidence * 100:.2f}%)"
        cv2.putText(frame, prediction_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), 2)
        cv2.putText(frame, f"Statement: {str}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), 2)



    # Display the resulting frame
    cv2.imshow("Data Collection", frame)

    # Exit on 'ESC' key press
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
