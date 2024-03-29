import cv2
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#imagePath = "3.jpg" 

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vc = cv2.VideoCapture(0) 

while True:
    _, img = vc.read() 
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = faceCascade.detectMultiScale(
    grayscale,
    scaleFactor=1.3,
    minNeighbors =5,
   minSize=(30, 30)
    )
    print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces found", img)
    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

vc.release() 
cv2.destroyAllWindows()


