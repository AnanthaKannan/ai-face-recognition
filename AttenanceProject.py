import cv2
import numpy as np
import face_recognition as fc
import os

path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodeings(images)
print('Encoding completed')


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrameLoc = fc.face_locations(imgS)
    encodeCurFrameLoc = fc.face_encodings(imgS, facesCurFrameLoc)

    for encodeFace, faceLoc in zip(encodeCurFrameLoc, facesCurFrameLoc):
        matches = fc.compare_faces(encodeListKnown, encodeFace)
        faceDis = fc.face_distance(encodeListKnown, encodeFace)
        # print(matches)
        matchIndex = np.argmin(faceDis)
        # print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            multiple = 4
            y1, x2, y2, x1 = y1 * multiple, x2 * multiple, y2 * multiple, x1 * multiple
            cv2.rectangle(img, (x1, y1), (x2, y2),(255, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)





