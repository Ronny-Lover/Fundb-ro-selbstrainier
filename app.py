import json
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = "model.h5"
LABELS_PATH = "labels.json"

# Diese Werte müssen zum Training deines Modells passen
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    image_array = np.array(image).astype("float32")

    # Häufige Normalisierung für CNN-Modelle
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def classify_image(model, image, labels):
    prepared_image = prepare_image(image)
    prediction = model.predict(prepared_image, verbose=0)

    prediction = np.array(prediction)

    # Klassisches Mehrklassenmodell mit Softmax
    if prediction.shape[-1] > 1:
        probabilities = prediction[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        category = labels[class_index]

    # Binäres Modell oder Sigmoid-Modell
    else:
        probability = float(prediction[0][0])

        if len(labels) == 2:
            class_index = 1 if probability >= 0.5 else 0
            confidence = probability if class_index == 1 else 1 - probability
            category = labels[class_index]
        else:
            category = labels[0]
            confidence = probability

    return category, confidence


def main():
    st.set_page_config(
        page_title="Digitales Fundbüro",
        page_icon="🔎",
        layout="centered"
    )

    st.title("🔎 Digitales Fundbüro")
    st.write(
        "Lade ein Bild eines gefundenen Gegenstands hoch. "
        "Die KI versucht, den Gegenstand zu kategorisieren."
    )

    try:
        model = load_model()
        labels = load_labels()
    except Exception as error:
        st.error("Das Modell oder die Kategorien konnten nicht geladen werden.")
        st.code(str(error))
        return

    uploaded_file = st.file_uploader(
        "Bild hochladen",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        st.info("Bitte lade ein Bild hoch.")
        return

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Hochgeladenes Bild",
        use_container_width=True
    )

    if st.button("Gegenstand analysieren", type="primary"):
        with st.spinner("Bild wird analysiert..."):
            try:
                category, confidence = classify_image(
                    model,
                    image,
                    labels
                )

                st.success(f"Kategorie: {category}")
                st.metric(
                    "Erkennungswahrscheinlichkeit",
                    f"{confidence * 100:.2f} %"
                )

                if confidence < 0.60:
                    st.warning(
                        "Die Erkennung ist unsicher. "
                        "Bitte überprüfe die Kategorie manuell."
                    )

                st.subheader("Fundstück speichern")

                location = st.text_input(
                    "Fundort",
                    placeholder="z. B. Bahnhof, Park oder Schule"
                )

                description = st.text_area(
                    "Zusätzliche Beschreibung",
                    placeholder="Farbe, Marke, besondere Merkmale ..."
                )

                contact = st.text_input(
                    "Kontaktmöglichkeit",
                    placeholder="E-Mail oder Telefonnummer"
                )

                if st.button("Fundstück speichern"):
                    record = {
                        "kategorie": category,
                        "vertrauen": round(confidence, 4),
                        "fundort": location,
                        "beschreibung": description,
                        "kontakt": contact,
                        "datum": datetime.now().isoformat()
                    }

                    with open(
                        "fundstuecke.json",
                        "a",
                        encoding="utf-8"
                    ) as file:
                        file.write(
                            json.dumps(
                                record,
                                ensure_ascii=False
                            ) + "\n"
                        )

                    st.success("Das Fundstück wurde gespeichert.")

            except Exception as error:
                st.error("Bei der Bildanalyse ist ein Fehler aufgetreten.")
                st.code(str(error))


if __name__ == "__main__":
    main()
