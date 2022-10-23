from ast import Bytes
from pydoc import importfile
from fastapi import FastAPI, File, UploadFile,Request
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st
import base64
import io

app = FastAPI()

MODEL = tf.keras.models.load_model("/home/a003k/Downloads/MLineuron/PROJECTS/Potato/models/potatoes.h5")
CLASS_NAMES = ["Early Blight","Late Blight", "Healthy"]

# @app.get("/hello/{name}")
# async def hello(name):
#     return f"welcome {name}"  

# def read_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...) # verify file as input
# ):
#     image = read_image(await file.read())    
#     image_batch = np.expand_dims(image, 0)
#     predictions= MODEL.predict(image_batch)
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
    
#     return {
#         'class' : predicted_class,
#         'confidence' : float(confidence)
#     }


@app.post("/predict")
async def predict(
    file: bytes = File(...) # verify file as input
):
    image =  np.array(Image.open(io.BytesIO(file)))
    image_batch = np.expand_dims(image, 0)
    predictions= MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }



if __name__== "__main__":
    uvicorn.run(app,host='0.0.0.0', port=8000)