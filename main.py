import cv2
import os
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml') 
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
#  detect face pic in folder
folderPath='./img/'
listFile=os.listdir(folderPath)
for file in listFile:
    imagePath=file
    print(file)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors =5,
    minSize=(30, 30)
    )
    print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
