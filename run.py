import numpy as np
from PIL import Image
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('Nariz.xml')


cap =  cv2.VideoCapture('Test1.mp4')

ImageAfterProcessing = []
frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #get videos frames count
ds_factor = 1.5
print(frame)
for i in range(frame):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)#detecting noses

        for (nx, ny, nw, nh) in nose:#draw  noses rectangle
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
        ImageAfterProcessing.append(img) #append the new frame with detected features

for frame in ImageAfterProcessing:
    cv2.imshow('frame', frame)
    if cv2.waitKey(35) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()