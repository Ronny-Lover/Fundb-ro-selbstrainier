import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Einstellungen
# --------------------------------------------------

st.set_page_config(
    page_title="Funbüro KI",
    page_icon="👕",
    layout="centered"
)

CLASS_NAMES = [
    "Hosen",
    "Jacken und hoodies",
    "Schuhe",
    "Tshirt"
]

MODEL_PATH = "model.h5"

# Hier musst du ggf. die Größe anpassen,
# mit der dein Modell trainiert wurde.
IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# Modell laden
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Oberfläche
# --------------------------------------------------

st.title("👕 Funbüro")
st.write("Lade ein Kleidungsstück hoch und die KI ordnet es einer Kategorie zu.")


uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Bild verarbeiten und vorhersagen
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Hochgeladenes Bild",
        use_container_width=True
    )

    if st.button("🔍 Bild analysieren"):

        with st.spinner("KI analysiert das Bild..."):

            # Bild auf Modellgröße bringen
            resized_image = image.resize(IMAGE_SIZE)

            # In NumPy-Array umwandeln
            image_array = np.array(resized_image)

            # Pixelwerte normalisieren
            image_array = image_array.astype("float32") / 255.0

            # Batch-Dimension hinzufügen
            image_array = np.expand_dims(image_array, axis=0)

            # Vorhersage
            prediction = model.predict(image_array)

            # Wahrscheinlichkeiten
            probabilities = prediction[0]

            # Index der höchsten Wahrscheinlichkeit
            predicted_index = np.argmax(probabilities)

            # Kategorie
            predicted_class = CLASS_NAMES[predicted_index]

            # Wahrscheinlichkeit
            confidence = probabilities[predicted_index] * 100

        st.success(f"Erkannte Kategorie: **{predicted_class}**")

        st.metric(
            "Treffsicherheit",
            f"{confidence:.2f} %"
        )

        # --------------------------------------------------
        # Wahrscheinlichkeiten anzeigen
        # --------------------------------------------------

        st.subheader("Wahrscheinlichkeiten")

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):
            st.write(
                f"**{class_name}:** "
                f"{probability * 100:.2f} %"
            )

            st.progress(
                float(probability)
            )
