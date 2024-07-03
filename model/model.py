from get_logger import *
import time 

start = time.time()
from keras.models import Model 
from keras.layers import LSTM, Input, Dense
from keras.models import load_model
logger.info(f"Keras loading time: {time.time()-start}")
from constants import *

def get_model(n_sign):
    input_shape = (FRAMES, FEATURES)

    # Define the input layer
    inputs = Input(shape=input_shape, name='input_layer')

    # LSTM layers
    lstm_1 = LSTM(units=50, return_sequences=False, return_state=False, name='lstm_1')(inputs)

    # Output layer
    outputs = Dense(units=n_sign, activation='softmax', name='output_layer')(lstm_1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Display the model summary
    model.summary()

    return model

def load_model_():
    return load_model("weights/trained.keras")
