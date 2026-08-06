# All Imports
import numpy as np
import pandas as pd
import cv2

from PIL import Image

import tensorflow as tf

from tensorflow.keras.models import load_model, Model

from tensorflow.keras.layers import (
    Input,
    Lambda,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0
)

import streamlit as st




# page config
st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="wide"
)

st.title("😊 DeepFER")
st.subheader("Facial Emotion Recognition")


# class labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


def build_mobilenet():
    ######mobilenetv2 load######
    NUM_CLASSES = 7

    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224,224,3)
    )

    inputs = Input(shape=(48,48,1))

    x = Lambda(
        lambda img: tf.image.grayscale_to_rgb(img),
        output_shape=(48,48,3)
    )(inputs)

    x = Lambda(
        lambda img: tf.image.resize(img,(224,224)),
        output_shape=(224,224,3)
    )(x)

    x = base_model(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    mobilenet_model = Model(inputs, outputs)

    return mobilenet_model

    # mobilenet_model.load_weights(
    #     "mobilenetv2_final_final_Hopefully_v8.keras"
    # )

    # print("MobileNet weights loaded.")

def build_efficientnet():
    ######efficientnet load######
    NUM_CLASSES = 7

    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224,224,3)
    )

    inputs = Input(shape=(48,48,1))

    x = Lambda(
        lambda img: tf.image.grayscale_to_rgb(img),
        output_shape=(48,48,3)
    )(inputs)

    x = Lambda(
        lambda img: tf.image.resize(img,(224,224)),
        output_shape=(224,224,3)
    )(x)

    x = base_model(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    efficientnet_model = Model(inputs, outputs)

    return efficientnet_model

    # efficientnet_model.load_weights(
    #     "efficientnetb0_final.keras"
    # )

    # print("EfficientNet weights loaded.")

@st.cache_resource
def load_selected_model(name):

    if name == "Baseline CNN":
        return load_model("models/baseline_cnn_final.keras")

    elif name == "Improved CNN (Weighted)":
        return load_model("models/improved_cnn_weighted_final.keras")

    elif name == "Improved CNN (Unweighted)":
        return load_model("models/improved_cnn_final_unweighted.keras")

    elif name == "MobileNetV2":
        model = build_mobilenet()
        model.load_weights(
            "models/mobilenetv2_final_final_Hopefully_v8.keras"
        )
        return model

    else:
        model = build_efficientnet()
        model.load_weights(
            "models/efficientnetb0_final.keras"
        )
        return model


def preprocess_image(image):

    gray = image.convert("L")

    img48 = np.array(gray)
    img48 = cv2.resize(img48,(48,48))
    img48 = img48.astype(np.float32)/255.0

    cnn_input = img48.reshape(1,48,48,1)

    return cnn_input, cnn_input.copy()



# sidebar
selected_model = st.sidebar.radio(

    "Choose Model",
    [
        "Improved CNN (Weighted)",
        "Baseline CNN",
        "Improved CNN (Unweighted)",
        "MobileNetV2",
        "EfficientNetB0"
    ]
)


# Upload widget
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)



# Display image
if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


# Predict button
if uploaded_file and st.button("Predict"):

    model = load_selected_model(selected_model)

    cnn_input, transfer_input = preprocess_image(image)
    pred = model.predict(
        transfer_input
        if "MobileNet" in selected_model or "EfficientNet" in selected_model
        else cnn_input,
        verbose=0
    )

    # prediction
    idx = np.argmax(pred)
    st.success(
        f"Prediction : {emotion_labels[idx]}"
    )

    st.write(
        f"Confidence : {pred[0][idx]*100:.2f}%"
    )

    # probability table
    df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Probability": pred[0]
        })

    # Convert probabilities to percentages
    df["Probability (%)"] = df["Probability"] * 100

    st.subheader("Probability Distribution")

    st.bar_chart(
        df.set_index("Emotion")["Probability (%)"]
    )

    st.dataframe(
        df[["Emotion", "Probability (%)"]].style.format({"Probability (%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )



