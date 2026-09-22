import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Reihenfolge muss der Reihenfolge beim Training entsprechen
CLASS_NAMES = [
    "Hosen",
    "Jacken/Hoodies",
    "Schuhe",
    "T-Shirts"
]

IMG_HEIGHT = 224
IMG_WIDTH = 224


# --------------------------------------------------
# Seitenkonfiguration
# --------------------------------------------------

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# Responsive CSS für Handy und PC
# --------------------------------------------------

st.markdown(
    """
    <style>
        /* Gesamter Hintergrund */
        .stApp {
            background-color: #f5f7fb;
        }

        /* Hauptbereich */
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 3rem;
        }

        /* Überschrift */
        .main-title {
            text-align: center;
            color: #1f2937;
            font-size: clamp(2rem, 6vw, 3.5rem);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: clamp(1rem, 3vw, 1.25rem);
            margin-bottom: 2rem;
        }

        /* Karten */
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 18px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .result-card {
            background-color: #ecfdf5;
            border-left: 6px solid #10b981;
            padding: 1.25rem;
            border-radius: 14px;
            margin-top: 1rem;
        }

        .result-title {
            color: #065f46;
            font-size: clamp(1.3rem, 4vw, 2rem);
            font-weight: bold;
        }

        .confidence {
            color: #047857;
            font-size: 1.1rem;
        }

        /* Buttons auf PC und Smartphone gut bedienbar */
        .stButton > button {
            width: 100%;
            min-height: 3.2rem;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 600;
        }

        /* Datei-Upload */
        [data-testid="stFileUploader"] {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 14px;
        }

        /* Bilder mobil nicht überlaufen lassen */
        img {
            max-width: 100%;
            height: auto;
            border-radius: 14px;
        }

        /* Abstand bei kleinen Displays */
        @media (max-width: 600px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }

            .card {
                padding: 1rem;
                border-radius: 14px;
            }

            .stButton > button {
                min-height: 3.5rem;
                font-size: 1.1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Modell laden
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")


# --------------------------------------------------
# Bild vorbereiten
# --------------------------------------------------

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    image_array = np.array(image).astype("float32")

    # Nur verwenden, wenn das Modell beim Training
    # ebenfalls mit Werten zwischen 0 und 1 trainiert wurde.
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# Vorhersage
# --------------------------------------------------

def predict_image(image):
    model = load_model()
    prepared_image = prepare_image(image)

    predictions = model.predict(prepared_image, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions


# --------------------------------------------------
# Kopfbereich
# --------------------------------------------------

st.markdown(
    '<div class="main-title">👕 Digitales Fundbüro</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Kleidungsstücke automatisch erkennen</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Upload-Bereich
# --------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Bild hinzufügen")

st.write(
    "Wähle ein Bild aus deiner Galerie aus oder fotografiere "
    "das Kleidungsstück direkt mit deinem Smartphone."
)

# Kamera für Smartphones
camera_image = st.camera_input("📷 Foto aufnehmen")

st.write("Oder ein vorhandenes Bild hochladen:")

uploaded_file = st.file_uploader(
    "Bild auswählen",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)


# Kamera oder Upload verwenden
selected_file = camera_image if camera_image is not None else uploaded_file


if selected_file is not None:
    image = Image.open(selected_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Vorschau")

    st.image(
        image,
        caption="Ausgewähltes Bild",
        use_container_width=True
    )

    recognize_button = st.button(
        "🔍 Kategorie erkennen",
        type="primary"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if recognize_button:
        try:
            with st.spinner("Das Bild wird analysiert ..."):
                predicted_class, confidence, predictions = predict_image(image)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        Erkannte Kategorie: {predicted_class}
                    </div>
                    <div class="confidence">
                        Sicherheit: {confidence * 100:.2f} %
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("Ergebnisübersicht")

            for class_name, probability in zip(CLASS_NAMES, predictions):
                percentage = float(probability) * 100

                st.write(f"**{class_name}**: {percentage:.2f} %")
                st.progress(min(float(probability), 1.0))

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as error:
            st.error(
                "Das Bild konnte nicht verarbeitet werden. "
                "Bitte überprüfe dein Modell und die Bildgröße."
            )

            with st.expander("Technische Fehlermeldung"):
                st.exception(error)


else:
    st.info(
        "Lade ein Bild hoch oder nimm ein Foto mit der Kamera auf, "
        "um eine Kategorie zu erkennen."
    )
