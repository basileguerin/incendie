import av
import base64
import cv2
import imageio.v3 as iio
import io
import numpy as np
import pytz
import streamlit as st
import tempfile
import torch
from datetime import datetime
from PIL import Image
from pymongo import MongoClient
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Chargement du modèle YOLOv5 entraîné pour la detection d'incendies
MODEL_PATH = 'yolov5/runs/train/incendiemodelv2/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# --- Base de données ---
db_client = MongoClient('localhost', 27018, username='root', password='1234')
db = db_client['incendies']
collections = {
    "Image": db["image_detection"],
    "Vidéo": db["video_detection"],
    "Webcam": db["webcam_detection"],
}

# Modification flux webcam grâce à streamlit_webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_frame = process_data(img, "Webcam", "webcam live")

        return av.VideoFrame.from_ndarray(processed_frame, format="rgb24")

def save_detection(option, detections, filename=None):
    """
    Fonction pour enregistrer les détections dans la base de données
    - `option` : option de détection (Image, Vidéo, Webcam)
    - `detections` : liste des détections dans une image ou frame
    - `filename` : nom du fichier

    Enregistre chaque détection avec date, coordonnées des bounding box, nom de
    la classe et le taux de confiance.
    """
        
    classes_dict = {
        0 : 'fire',
        1 : 'smoke'
    }

    # Convertit la date actuelle au format heure française
    tz = pytz.timezone('Europe/Paris')
    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d %H:%M")

    collection = collections[option]

    for box, conf, cls in detections:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        data = {
            "filename" : filename,
            "date": date_str,
            "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "class": classes_dict[int(cls)],
            "confidence": float(conf),
        }
        collection.insert_one(data)

def display_detections(option):
    """
    Fonction pour afficher les détections à partir de la base de données
    - `option` : option de détection (Image, Vidéo, Webcam)

    Ecrit le contenu de la collection via streamlit.
    """
    collection = collections[option]
    detections = collection.find()

    # Bouton pour vider la collection
    nb_detections = collection.count_documents({})
    if nb_detections > 0:
        if st.button("Vider la collection"):
            clear_collection(option)
            st.write("La collection a été vidée.")
    else:
        st.write("La collection est vide.")

    for detection in detections:
        st.write(f"Nom du fichier: {detection['filename']}")
        st.write(f"Date: {detection['date']}")
        st.write(f"Coordonnées: {detection['coordinates']}")
        st.write(f"Classe: {detection['class']}")
        st.write("--"*10)

def clear_collection(option):
    """
    Fonction pour supprimer tous les documents d'une collection
    - `option` : option de détection (Image, Vidéo, Webcam)
    """
    collection = collections[option]
    collection.delete_many({})

def process_data(image, data_type, filename):
    """
    Fonction qui add les bounding box à une image et appelle `save_detection`
    pour enregistrer en base.

    - `image` : image ou frame à traiter
    - `data_type` : "Image", "Vidéo" ou "Webcam"
    - `filename` : nom du fichier
    """
    results = model(image)
    detections = []

    for *box, conf, cls in results.xyxy[0]:
        detections.append((box, conf, cls))

    save_detection(data_type, detections, filename)
    return results.render()[0]

def img_to_base64(img, format="JPEG"):
    """
    Conversion d'une image en base64 pour permettre son téléchargement.
    - `img`: Image à convertir
    - `format`: format de sortie (default : JPEG)
    """
    img_pil = Image.fromarray(img)
    buffer = io.BytesIO()
    img_pil.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode()

def video_to_base64(video_path):
    """
    Conversion d'une vidéo en base64 pour permettre son téléchargement.
    - `video_path`: chemin vers la vidéo
    """
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode()

def process_image_option():
    """
    Fonction pour gérer la partie image de l'application streamlit.
    """

    # Charger l'image
    uploaded_image = st.file_uploader("Choisissez une image", 
                                      type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Convertir l'image en tableau numpy et traiter avec le modèle
        input_image = Image.open(uploaded_image)
        input_image_np = np.array(input_image)
        processed_image_np = process_data(input_image_np, "Image", 
                                          uploaded_image.name)

        # Afficher les images avant et après le traitement
        st.subheader("Image originale")
        st.image(input_image, use_column_width=True)
        st.subheader("Image traitée avec détection d'incendie")
        st.image(processed_image_np, use_column_width=True)
    
        if st.button("Télécharger l'image traitée"):
            img_base64 = img_to_base64(processed_image_np)
            href = f'<a href="data:image/jpeg;base64,{img_base64}" download="processed_image.jpg">Télécharger l\'image traitée</a>'
            st.markdown(href, unsafe_allow_html=True)

def process_video_option():
    """
    Fonction pour gérer la partie vidéo de l'application streamlit.
    """
    
    # Charger la vidéo
    uploaded_file = st.file_uploader("Choisissez une vidéo", 
                                     type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Lecture de la vidéo
        cap = cv2.VideoCapture(tfile.name)

        # Vérifier si la vidéo est chargée correctement
        if not cap.isOpened():
            st.write("Erreur lors de l'ouverture de la vidéo.")
        else:
            st.write("Vidéo chargée avec succès.")

            # Obtenir les dimensions de la vidéo
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Créer un objet VideoWriter pour écrire la vidéo de sortie
            output_file = tempfile.NamedTemporaryFile(delete=False, 
                                                      suffix=".mp4")
            out = cv2.VideoWriter(output_file.name, 
                                  cv2.VideoWriter_fourcc(*'mp4v'), 
                                  30, (width, height))

            st.write("Traitement de la vidéo...")

            # Boucle sur les images de la vidéo
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Traiter l'image avec le modèle YOLOv5 et obtenir les boîtes englobantes
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_data(frame, "Vidéo", 
                                               uploaded_file.name)

                # Enregistrer l'image traitée dans la vidéo de sortie
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

            # Libérer les ressources
            cap.release()
            out.release()

            # Convertir la vidéo en WebM
            output_webm_file = tempfile.NamedTemporaryFile(delete=False, 
                                                           suffix=".webm")
            input_video = iio.imread(output_file.name)
            iio.imwrite(output_webm_file.name, input_video, fps=30, 
                        codec="libvpx-vp9")

            st.write("Vidéo traitée avec succès.")
            st.video(output_webm_file.name)

            if st.button("Télécharger la vidéo traitée"):
                video_base64 = video_to_base64(output_webm_file.name)
                href = f'<a href="data:video/webm;base64,{video_base64}" download="processed_video.webm">Télécharger la vidéo traitée</a>'
                st.markdown(href, unsafe_allow_html=True)

def process_webcam():
    """
    Fonction pour gérer la partie webcam de l'application streamlit.
    """
    st.header("Webcam Live Stream")
    webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

# --- Application ---
st.set_page_config(page_title="Détection d'indendies", page_icon=":fire:")
st.title("Détection d'incendie")

# Choix de la source d'entrée
option = st.sidebar.selectbox(
    "Choisissez la source",
    ("Image", "Vidéo", "Webcam", "Afficher les détections")
)

if option == "Image":
    process_image_option()

elif option == "Vidéo":
    process_video_option()

elif option == "Webcam":
    process_webcam()

else:
    display_option = st.selectbox(
    "Sélectionnez l'option à afficher",
    ("Image", "Vidéo", "Webcam"),
)
    display_detections(display_option)