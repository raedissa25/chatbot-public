import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from tensorflow.keras.models import load_model
import streamlit as st
api_key = st.secrets["GEMINI_API_KEY"]
if api_key is None:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")
genai.configure(api_key=api_key)

# Configuration Gemini
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

# Charger le modèle CNN
MODEL_PATH = "models/resnet50v2_ecg_best_model.h5"
resnet50v2_model = load_model(MODEL_PATH)

# Dictionnaire des classes
classes = {
    0: ("Normal (N)", "Battement cardiaque normal (rythme sinusal normal)."),
    1: ("Extrasystole ventriculaire (VEB / PVC)", "Battement prématuré provenant des ventricules."),
    2: ("Battements supraventriculaires (SVEB)", "Battements prématurés provenant des oreillettes."),
    3: ("Fusion de battements (F)", "Fusion entre un battement normal et un VEB."),
    4: ("Battements inconnus (Q)", "Battements non classifiés dans les catégories précédentes."),
}

# Prétraitement des données ECG
def preprocess_ecg(csv_file):
    df = pd.read_csv(csv_file, header=None)
    if df.shape[1] != 188:
        st.error("Le fichier CSV doit contenir exactement 188 colonnes.")
        return None
    ecg_signal = df.iloc[:, :-1].values  # shape (samples, 187)
    ecg_data = np.expand_dims(ecg_signal, axis=-1)  # shape (samples, 187, 1)
    return ecg_data

# Tracer un signal ECG
def plot_ecg_signal(signal, title="ECG Signal (1er échantillon)"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, color='dodgerblue')
    ax.set_title(title)
    ax.set_xlabel("Temps (échantillons)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

# Obtenir réponse IA
def get_bot_response(user_input):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_input)
    return response.text

# App Streamlit
def main():
    st.title("🫀 Assistant Médical Cardiaque")

    # Upload du fichier ECG
    st.header("📂 téléverser un fichier CSV ECG pour l'analyse")
    uploaded_file = st.file_uploader("téléverser un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        st.success("✅ Fichier uploadé avec succès !")

        ecg_input = preprocess_ecg(uploaded_file)
        if ecg_input is not None:
            st.subheader("📈 Aperçu du signal ECG")
            plot_ecg_signal(ecg_input[0].squeeze())

            # Prédiction CNN
            prediction = resnet50v2_model.predict(ecg_input)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            class_name, class_description = classes[predicted_class_idx]

            st.subheader("✅ Résultat de l'analyse du signal ECG :")
            st.success(f"Classe prédite : **{class_name}**")
            st.markdown(f"🔎 *{class_description}*")

            st.subheader("📊 Détail des probabilités pour chaque classe :")
            for idx, prob in enumerate(prediction[0]):
                name, desc = classes[idx]
                st.markdown(f"**{name}** — {desc} : **{prob:.4f}**")
                st.progress(float(prob))

    # Section Chat IA
    st.header("💬 Posez une question à votre cardiologue IA")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Posez une question sur le cœur, les anomalies, etc.")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Réflexion en cours..."):
            bot_response = get_bot_response(user_input)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == '__main__':
    main()
