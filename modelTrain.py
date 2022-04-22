import cv2
import numpy as np
from PIL import Image
import os

path = 'sample'
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def imagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        # convet image to grayscale
        gray_img = Image.open(imagePath).convert('L')
        img_arr = np.array(gray_img, 'uint8')

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_arr)

        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y : y + h, x : x + w])
            Ids.append(Id)

    return faceSamples, Ids

print("Training face. wait for few seconds...")

faces, Ids = imagesAndLabels(path)
recognizer.train(faces, np.array(Ids))

# check trainer folder exist or not if not create one
if (os.path.isdir('trainer/') != True):
    os.mkdir('trainer')

# saving the trained model file
recognizer.write('trainer/trainer.yml')

print("Model training complete.")