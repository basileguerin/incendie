# Transfer Learning yolov5

Appliquer le transfer learning sur une base de données. 

Apprentissage supervisé pour détecter les incendies à partir d'une vidéo ou image

L'entraînement du modèle yolov5s a été effectué via colab avec le notebook suivant :

https://colab.research.google.com/drive/1J1GIsYUDuJxeIcgYKRFtmHfJ4PY7hEaM#scrollTo=0tVWRVzf9iGb

Les différents modèles sont présents dans yolov5/runs/train :
- incendiemodelv2 : yolov5s
- incendiemodelv3 : yolov5x

Les performances des deux modèles étant similaires on utilisera incendiemodelv2 dans l'application pour des runtime plus courts.

# Application

Application Streamlit qui est capable de :

- Charger et executer la détection d'image, vidéo ou webcam
- Stocker les détections en base de données

Le code de l'application est contenu dans le fichier `app.py`.

Vidéo qui montre le fonctionnement de l'application : `vidéo démo.mkv`