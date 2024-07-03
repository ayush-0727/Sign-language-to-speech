import cv2 
from utils.mediapipe_helper import MediapipeHelper
from constants import * 
import numpy as np
import os

cap = cv2.VideoCapture(0)
mediapipeHelper = MediapipeHelper()

while True:
    sign = input("Enter sign name: ")
    
    if(sign == "q"):
        cv2.destroyAllWindows()
        cap.release
        break

    # Specify the directory where you want to save the video
    output_directory = 'Videos/'  # Replace with your desired folder path
    output_filename = sign + '.avi'
    output_path = os.path.join(output_directory, output_filename)

    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs as well (e.g., 'MJPG', 'X264', etc.)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    ## collect SAMPLES of the sign
    for i in range(SAMPLES):
        data_frames = []
        ctr = 0
        while ctr < FRAMES:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            features, frame = mediapipeHelper.getFeatures(frame)
            if(len(features) != 0):
                ctr += 1
                data_frames.append(features)

            cv2.putText(frame, f"HANDS: {ctr}", (20,55), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,(0,0,255))
            cv2.putText(frame, f"ROUND: {i+1}", (20,75), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,(0,0,255))
            cv2.imshow("Data Collection", frame)

            # Write the frame into the file
            if i<5:
                out.write(frame)

            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                cap.release()
                break

        data_frames = np.array(data_frames)
        file_path = f"{DATASET_DIR}/{sign}_{i}.npy"
        np.save(file_path, data_frames)
        print(f"Dataset saved at --> {file_path} of shape --> {data_frames.shape}")

