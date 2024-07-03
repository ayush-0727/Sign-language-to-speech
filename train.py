from constants import * 
from model.model import get_model, logger
from data_loader import load_data
import time

start = time.time()
## read data from /dataset and send out X->(BATCH, FRAMES, FEATURES), y->labels
X, y, n_sign = load_data()
logger.info(f"Loading dataset took : {time.time() - start}")

model = get_model(n_sign)

logger.info(f"Training model on dataset --> {X.shape} and {y.shape}")
history = model.fit(X, y, epochs=50)

model.save("./weights/trained.keras", save_format='tf')