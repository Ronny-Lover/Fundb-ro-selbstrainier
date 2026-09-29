import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Konfiguration
# --------------------------------------------------

st.set_page_config(
    page_title="Fundbüro",
    page_icon="🔎",
    layout="centered"
)


# --------------------------------------------------
# Kategorien
# --------------------------------------------------

CATEGORIES = [
    "Hosen",
    "Jacken und Hoodies",
    "Schuhe",
    "T-Shirt"
]


# --------------------------------------------------
# Modell laden
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")


model = load_model()


# --------------------------------------------------
# Bildgröße aus Modell auslesen
# --------------------------------------------------

def get_image_size():

    input_shape = model.input_shape

    height = input_shape[1]
    width = input_shape[2]

    if height is None or width is None:
        raise ValueError(
            "Die Eingabegröße des Modells konnte nicht ermittelt werden."
        )

    return int(width), int(height)


# --------------------------------------------------
# Bild klassifizieren
# --------------------------------------------------

def predict_image(image):

    width, height = get_image_size()

    # RGB erzwingen
    image = image.convert("RGB")

    # Auf Modellgröße bringen
    image = image.resize((width, height))

    # NumPy
    image_array = np.array(image)

    # Normalisierung
    image_array = image_array.astype("float32") / 255.0

    # Batch Dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Vorhersage
    prediction = model.predict(
        image_array,
        verbose=0
    )

    probabilities = prediction[0]

    predicted_index = np.argmax(probabilities)

    category = CATEGORIES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    return category, confidence, probabilities


# --------------------------------------------------
# Benutzeroberfläche
# --------------------------------------------------

st.title("🔎 Fundbüro")

st.write(
    "Lade ein Bild eines gefundenen Kleidungsstücks hoch. "
    "Das KI-Modell versucht anschließend, die Kategorie zu erkennen."
)


uploaded_file = st.file_uploader(
    "Bild auswählen",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Hochgeladenes Bild",
        use_container_width=True
    )

    if st.button(
        "🔍 Bild analysieren",
        use_container_width=True
    ):

        try:

            category, confidence, probabilities = predict_image(image)

            st.success(
                f"Erkannt: **{category}**"
            )

            st.metric(
                "Konfidenz",
                f"{confidence * 100:.2f} %"
            )

            # Wahrscheinlichkeiten anzeigen
            st.subheader("Wahrscheinlichkeiten")

            for category_name, probability in zip(
                CATEGORIES,
                probabilities
            ):

                st.write(
                    f"**{category_name}**: "
                    f"{probability * 100:.2f} %"
                )

                st.progress(
                    float(probability)
                )

        except Exception as e:

            st.error(
                f"Fehler bei der Bilderkennung: {e}"
            )


# --------------------------------------------------
# Informationen
# --------------------------------------------------

st.divider()

st.subheader("Erkennbare Kategorien")

for category in CATEGORIES:
    st.write(f"- {category}")

