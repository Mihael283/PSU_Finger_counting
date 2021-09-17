import cv2
import mediapipe as mp


mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45 #Koordinate za print fukciju 

tipIds = [4, 8, 12, 16, 20] #Oznaka vrha prsta

video = cv2.VideoCapture(0)

hands = mp_hand.Hands(max_num_hands=1) #Broj ruke 

while True:
    ret, image = video.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lmList = []  
    if results.multi_hand_landmarks:                                   
        for hand_landmark in results.multi_hand_landmarks:    #Petlja za pracenje ruke
            myHands = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

            mp_draw.draw_landmarks(
                image, hand_landmark,
                mp_hand.HAND_CONNECTIONS)

    fingers = []
    if len(lmList) != 0:
        # Palac
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)  #DA
        else:
            fingers.append(0)  #NE

        # Ostali prsti
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)  #DA
            else:
                fingers.append(0)  #NE

        total = fingers.count(1)   
        cv2.putText(image, 'Prediction: %s' % (total), (fx,fy), font, 1.0, (245,210,65), 2, 1)

    cv2.namedWindow("Finger Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Finger Counter",720,480)
    imS = cv2.resize(image, (720, 480))       
    cv2.imshow("Finger Counter", imS)    

  

    k = cv2.waitKey(1)
    #press q za close
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()