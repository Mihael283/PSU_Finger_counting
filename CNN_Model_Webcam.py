from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
import os


dataColor = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45 #pozicija print funkcije u windowu
className = '0'
showMask = 0


classes = '0 1 2 3 4 5'.split()


def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


def main():
    global font, fx, fy, fh
    global dataColor
    global className
    global showMask

    model = load_model('model_v3.h5')

    x0, y0, width = 200, 140, 300

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    while True:
        # Frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0,y0), (x0+width-1,y0+width-1), dataColor, 12)

        #Crveni kvadrat oznacava region of interest
        roi = frame[y0:y0+width,x0:x0+width]
        roi = binaryMask(roi)

        #Pokazi masku samo u zadanoj regiji
        if showMask:
            window[y0:y0+width,x0:x0+width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        #predict
        img = np.float32(roi)/255.
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        img = np.resize(img, (4,300, 300,1))
        pred = classes[np.argmax(model.predict(img)[0])]
        cv2.putText(window, 'Prediction: %s' % (pred), (fx,fy), font, 1.0, (245,210,65), 2, 1)

        #window
        cv2.imshow('Original', window)

        key = cv2.waitKey(10) & 0xff
        # press q za close
        if key == ord('q'):
            break
        elif key == ord('b'):
            showMask = not showMask

    cam.release()


if __name__ == '__main__':
    main()