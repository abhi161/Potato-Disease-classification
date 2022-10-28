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

        api ="http://18.212.31.155/predict"
        m = MultipartEncoder(fields={'file':('filename', image_file ,'image/jpg')})

        response = requests.post(api, data=m, headers={'Content-Type':m.content_type})
        pred = response.json()

        col1,col2 = st.columns(2)
        with col1:
            st.image(image_file)

        with col2:
            st.write(pred['class'])
            st.write(pred['confidence'])

            if pred['class'] == 'Early Blight':
                st.subheader("Treatement suggestions")
                st.write("Prune or stake plants to improve air circulation and reduce fungal problems.")
                st.write("Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut")
                st.write("For best control, apply copper-based fungicides early, two weeks before disease normally appears or when weather forecasts predict a long period of wet weather. Alternatively, begin treatment when disease first appears, and repeat every 7-10 days for as long as needed.")
            else:
                st.subheader("Treatement suggestions")
                st.write("Apply a copper based fungicide (2 oz/ gallon of water) every 7 days or less, following heavy rain or when the amount of disease is increasing rapidly. If possible, time applications so that at least 12 hours of dry weather follows application.")
                st.write("Used as a foliar spray, Organocide® Plant Doctor will work its way through the entire plant to prevent fungal problems from occurring and attack existing many problems. Mix 2 tsp/ gallon of water and spray at transplant or when direct seeded crops are at 2-4 true leaf, then at 1-2 week intervals as required to control disease.")
                st.write("")




#if __name__ == "__main__":
