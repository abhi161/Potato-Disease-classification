from fastapi import File
from matplotlib.pyplot import imshow
import streamlit as st
import numpy as np
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64
import json                    


st.title("Potato Disease identifier")
image_file = st.file_uploader("Drop the image")

if st.button("predict"):

    if image_file is not None:

        api ="http://0.0.0.0:8000/predict"
        m = MultipartEncoder(fields={'file':('filename', image_file ,'image/jpg')})

        response = requests.post(api, data=m, headers={'Content-Type':m.content_type})
        pred = response.json()

        col1,col2 = st.columns(2)
        with col1:
            st.image(image_file)

        with col2:
            st.write(pred['class'])
            st.write(pred['confidence'])
#if __name__ == "__main__":
