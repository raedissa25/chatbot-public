import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import google.generativeai as genai

# Clé API Gemini
api_key = st.secrets["GEMINI_API_KEY"]
if api_key is None:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")
genai.configure(api_key=api_key)

# Configuration du modèle Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Charger le modèle ResNet50V2
MODEL_PATH = "resnet50v2_ecg_best_model.h5"
resnet_model = load_model(MODEL_PATH)

# Dictionnaire des classes
classes = {
    0: ("AHB", "Anomalie cardiaque non spécifique."),
    1: ("HMI", "Antécédent d'infarctus du myocarde."),
    2: ("MI", "Infarctus du myocarde actif."),
    3: ("Normal", "Rythme cardiaque normal."),
}

# Prétraitement de l'image ECG
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # (1, 224, 224, 3)
    return image_array

# Prédiction de classe
def predict_class(image):
    processed = preprocess_image(image)
    prediction = resnet_model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name, class_description = classes[predicted_class]
    return class_name, class_description, prediction[0]

# Générer réponse IA Gemini
def get_bot_response(user_input):
    chat = model.start_chat(history=[])
    response = chat.send_message(user_input)
    return response.text

# App Streamlit
def main():
    st.set_page_config(page_title="ECG Image Classifier", layout="centered")
    st.title("🫀 Diagnostic d’Image ECG via ResNet50V2")

    st.header("📷 Téléversez une image ECG (JPG/PNG)")
    uploaded_image = st.file_uploader("Choisir une image ECG", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image, caption="🖼️ Image ECG fournie", use_column_width=True)
        image = Image.open(uploaded_image).convert("RGB")

        class_name, class_description, probs = predict_class(image)

        st.subheader("✅ Résultat de l’analyse :")
        st.success(f"Classe prédite : **{class_name}**")
        st.markdown(f"🔍 *{class_description}*")

        st.subheader("📊 Détail des probabilités :")
        for idx, prob in enumerate(probs):
            name, desc = classes[idx]
            st.markdown(f"**{name}** — {desc} : **{prob:.4f}**")
            st.progress(float(prob))

    st.header("💬 Chat IA Médical (Gemini)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Posez une question sur l’ECG, l’analyse, etc.")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Analyse en cours..."):
            bot_response = get_bot_response(user_input)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == '__main__':
    main()
