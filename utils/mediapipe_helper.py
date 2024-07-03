import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
from constants import *

class MediapipeHelper:
    def __init__(self):
        model_path = "weights/hand_landmarker.task"
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def getFeatures(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ## convert frame into mediapipe format
        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame,
        )

        ## detect and get hand landmarks
        hand_landmarks = self.landmarker.detect(
            mp_frame,
        ).hand_landmarks

        cv2.putText(frame, f"HANDS: {len(hand_landmarks)}", (20,25), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,(0,0,255))
        ## if no hand detected just return
        if len(hand_landmarks) == 0:
            return [], frame

        landmark_prep = []
    
        for hand in hand_landmarks:
            ## this is for drawing on frame
            landmark_pb2_ = []

            ## taking center poistion to normalise
            center_x, center_y = hand[13].x, hand[13].y
            ## extract the keypoints
            for landmark in hand:
                landmark_pb2_.append(landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    ))
                landmark_prep.extend([landmark.x - center_x, landmark.y - center_y])

            ## define proto
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(landmark_pb2_)

            ## draw the landmarks
            self.drawing_utils.draw_landmarks(
                frame, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if len(hand_landmarks) != 2:
            landmark_prep.extend([0]*42)

        return (landmark_prep, frame)
