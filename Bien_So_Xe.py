import cv2
import numpy as np
import matplotlib.pyplot as plt

cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

cars = cv2.imread('many_car.png')

gray =  cv2.cvtColor(cars, cv2.COLOR_BGR2GRAY)

def converToRGB(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

cars_detected = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20,20))

print('Tim bien so', len(cars_detected))

for (x,y,w,h) in cars_detected:
    cv2.rectangle(cars, (x,y), (x+w, y+h), (145, 60, 255), 5)

plt.imshow(converToRGB(cars))
plt.imsave('Detected_cars.png', converToRGB(cars))
plt.waitforbuttonpress
