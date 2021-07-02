import cv2
import numpy as np
import face_recognition as fc

imgElon = fc.load_image_file('images/elonMusk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgElonTest = fc.load_image_file('images/elonTest.jpg')
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

faceLoc = fc.face_locations(imgElon)[0]
encodeEln = fc.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = fc.face_locations(imgElonTest)[0]
encodeElnTest = fc.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

result = fc.compare_faces([encodeEln], encodeElnTest)
faceDis = fc.face_distance([encodeEln], encodeElnTest)
print(result, faceDis)
cv2.putText(imgElonTest, f'{result} {round(faceDis[0], 2)}', (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon', imgElon)
cv2.imshow('ElonTest', imgElonTest)
cv2.waitKey(0)

