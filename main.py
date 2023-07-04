# importing necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initializing mediapipe
mpHands = mp.solutions.hands  # performs the hand recognition algo
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils  # to draw detected key points

# load gesture recognizer model
model = load_model('mp_hand_gesture')

# load class names
f = open('gesture.names', 'r')
labels = f.read().split('\n')
f.close()
print(labels)

# initialize webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read() # read each frame
    x, y, z = frame.shape

    frame = cv2.flip(frame, 1)  # flip frame vertically
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # mediapipe works with RGB but opencv works with BGR format, so converting color format

    # get hand landmark prediction
    result = hands.process(framergb)

    class_name = ''

    if result.multi_hand_landmarks:
        keypts = []
        for handskps in result.multi_hand_landmarks:
            for kp in handskps.landmark:
                #print(id, kp)
                kpx = int(kp.x * x)  # width
                kpy = int(kp.y * y)  # height
                keypts.append([kpx, kpy])

            # draw key points on frames
            mpDraw.draw_landmarks(frame, handskps, mpHands.HAND_CONNECTIONS)

            #predict hand gestures
            prediction = model.predict([keypts])
            class_id = np.argmax(prediction)
            class_name = labels[class_id].capitalize()

    # show prediction on frame
    cv2.putText(frame, class_name, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)


    # show final output
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release webcam and destroy active windows
cap.release()
cv2.destroyAllWindows()