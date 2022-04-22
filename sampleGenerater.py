import cv2
import os

# capture videos through webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# set video frame-width
cam.set(3, 640)
# set video frame-height
cam.set(4, 480)


# haarcascade classifier is use to detect object
face_cascader = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# check sample folder exist or not if not create one
if (os.path.isdir('sample/') != True):
    os.mkdir('sample')

face_id = input("Enter a Numeric User Id: ")
print("Taking samples, look at camera...")
count = 0

while True:

    # reading frames
    ret, img = cam.read()
    # convert input images into grayscale for easy read
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascader.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        # draw ractangle on faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # to capture & save images into the datasets folder
        cv2.imwrite("sample/face." + str(face_id) + '.' + str(count) + ".jpg", gray_img[y : y + h, x : x + w])

        
        # display image in a windows
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    # if k = esc stop the program
    if k == 27:
        break
    # if sample size exceed("count") stop the program
    elif count >= 30:
        break

print("samples taken comlete...")
cam.release()
cv2.destroyAllWindows()