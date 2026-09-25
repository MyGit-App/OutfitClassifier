
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import os
MODEL_PATH = "covid_vgg16_model.keras"

from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

@st.cache_resource

def load_covid_model():
    
    model_path = hf_hub_download(
                        repo_id="brijesh-singh/OutfitClassifier",
                        filename="best_tuned_vgg16.keras",
                        token=st.secrets["HF_TOKEN"])
    return load_model(model_path)

model = load_covid_model()

classes = ["Blazer",
            "Dress",
            "Hat",
            "Hoodie", 
            "Longsleeve", 
            "Outwear", 
            "Pants", 
            "Polo", 
            "Shirt", 
            "Shoes", 
            "Shorts", 
            "Skirt", 
            "T-Shirt", 
            "Undershirt"
            ]


#st.image("banner.png")

uploaded_file = st.file_uploader(
    "Upload Outfit Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded outfit image",
        use_container_width=True
    )

    image = image.resize((128,128))

    image = np.array(image)/255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    pred = model.predict(image)

    class_index = np.argmax(pred)

    confidence = np.max(pred)*100

    st.success(
        f"Prediction : {classes[class_index]}"
    )

    st.info(
        f"Confidence : {confidence:.2f}%"
    )