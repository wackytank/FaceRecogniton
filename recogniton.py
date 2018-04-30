import cv2
import numpy as np

# Recognizer Object
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")  #Load training data

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)

fontface = cv2.FONT_ITALIC
fontscale = 1
fontcolor = (255, 255, 550)



while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf>50):
            if(Id==1):
                pos=0
                Id="Vaibhav"
            elif(Id==2):
                pos=1
                Id="Obama"
            elif(Id==3):
                pos=2
                Id="Bebe Rexha"
            elif(Id==4):
                Id="Bharat"
        else:
            Id="Unknown"
        cv2.putText(im, str(Id), (x, y + h), fontface, fontscale, fontcolor, lineType=cv2.LINE_AA, thickness=2)
    cv2.imshow('im',im)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
