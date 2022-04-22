import cv2
from cv2 import CAP_DSHOW
from cv2 import COLOR_BGR2GRAY

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

# number of persons you want to recognize
Id = 2

# leave first empty because counter starts from 0
names = ['', 'Ishan']

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# set video frame width and height
cam.set(3, 640)
cam.set(4, 480)

# set min window size
min_width = 0.1*cam.get(3)
min_height = 0.1*cam.get(4)

while True:
    ret, img = cam.read()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(min_width), int(min_height)),)

    for(x, y, w, h) in faces:
        # draw rectangle on face in image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # to predict on every single image
        Id, accuracy = recognizer.predict(gray_img[y : y + h, x : x + w])

        # Check if accuracy is less then 100 ==> "0" is perfect match
        if (accuracy < 100):
            Id = names[Id]
            accuracy = " {0}%".format(round(100 - accuracy))

        else:
            Id = "Unknown"
            accuracy = " {0}%".format(round(100 - accuracy))

        cv2.putText(img, str(Id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("closeing program")
cam.release()
cv2.destroyAllWindows()