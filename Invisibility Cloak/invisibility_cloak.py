import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow('hsv', hsv)

        # lower = hue-10, 100, 100 and higher = hue+10, 255, 255
        red = np.uint8([[[0, 0, 255]]])
        hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        #print(hsv_red) # [0, 255, 255]

        # Threshold the hsv value to get red color hsv
        lower_red = np.array([10, 100, 100])
        upper_red = np.array([179, 255, 255])
        # Lower limit : (0, 100, 100) to (10, 255, 255) and upper limit: (160, 100, 100) to(179, 255, 255).

        mask = cv2.inRange(hsv, lower_red, upper_red)
        # cv2.imshow('mask', mask)

        part1 = cv2.bitwise_and(back, back, mask=mask)
        #cv2.imshow('part1', part1)  

        mask = cv2.bitwise_not(mask)    

        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        #cv2.imshow('mask', part2)

        cv2.imshow('cloak', part1+part2)

        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()