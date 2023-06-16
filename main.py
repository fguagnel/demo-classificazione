from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import time
import tensorflow as tf

app = FastAPI()

#origins = [
#    "http://localhost:8080",    # Chiamante
#    "http://localhost:3000",    # Destinatario
#]

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# MICRO SERVIZIO PER VERIFICARE SE IL SERVER E' IN ESECUZIONE
# http://localhost:8000/ping
@app.get("/ping")
async def ping():
    return "Ciao, sono in esecuzione!"


# FUNZIONE CHE PERMETTE DI TRASFORMARE IL FILE IMMAGINE COME ARRAT NUMPY
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

#Peronospora
CLASS_NAMES = ["Inizale", "Avanzata", "Sana"]

MODEL = tf.keras.models.load_model("potatoes.h5")

# MICRO SERVIZIO PER EFFETTUARE L'UPLOAD DEL FILE
@app.post("/predict")
async def predict( file: UploadFile = File(...) ):
    
    milli_sec_inizio = int(round(time.time() * 1000))
    print('> INIZIO /predict')

    #model = getSingletonModel()

    #global model
    #if model is None:
    #    print('    > Recupero modello...')
    #    model = tf.keras.models.load_model("potatoes.h5")
    #    print('    > Modello recuperato!')


    print('    > Elaborazione immagine...')

    image = np.array(
        Image.open( BytesIO(await file.read()) ).convert("RGB").resize((256, 256))
    )

    image = image/255 # Normalizzazione immagine nel range da 0 a 1

    img_batch = np.expand_dims(image, 0)
    
    print('    > Elaborazione immagine. Effettuata!')

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print('  > Predizione: ', predicted_class)
    print('  > confidenza: ', confidence)

    milli_sec_fine = int(round(time.time() * 1000))
    print(f'> FINE   /predict {milli_sec_fine-milli_sec_inizio} ms')

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

