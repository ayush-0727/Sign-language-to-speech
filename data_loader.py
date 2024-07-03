import numpy as np 
import os
from constants import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from get_logger import *
import pickle as pkl

def load_data():
    X = []
    y = []

    for file in os.listdir(DATASET_DIR):
        file_path = os.path.join(DATASET_DIR, file)
        sign_name = file.split("_")[0]
        X.append(np.load(file_path))
        y.append(sign_name)

    n_sign = len(set(y))
    X = np.array(X)
    y = np.array(y)
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    logger.info(f"Lables are : {le.classes_}")
    classes = {idx: value for idx, value in enumerate(le.classes_)}
    with open(os.path.join(WEIGHTS_DIR, "classes.pkl"), "wb") as f:
        pkl.dump(classes, f)

    y = to_categorical(y)

    return (X, y, n_sign)

