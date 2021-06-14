import cv2
import datetime

img_path = "/home/pi/Downloads/photo/p1.jpeg"

face_cascade = cv2.CascadeClassifier("/home/pi/OpenCV_Python/haarcascade_frontalface_default.xml")

image = cv2.imread(img_path)

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

dt = str(datetime.datetime.now())

for x,y,w,h in faces:
    
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0,255,0), thickness=2)

cv2.imshow('output', image)

cv2.waitKey(0)

