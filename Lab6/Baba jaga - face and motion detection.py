"""
Małgorzata Cichowlas s16512
Before use: pip install numpy, pip install opencv-python
The goal of this program is to detect motion from the video capturing and to recognize a human's face.
Input: Video capturing from camera.
Output: Life-time detecting motion and recognizing face.
Based on tutorials and websites: https://github.com/opencv/opencv/tree/master/data/haarcascades , https://github.com/ahadcove/Motion-Detection
and code from NAI lecture.
"""

import cv2
import numpy as np

# Load and check face cascade
face_cascade = cv2.CascadeClassifier(
    'C:\Program Files\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

# Video capturing from camera
cap = cv2.VideoCapture(0)

# Use subtractor to compare foreground with backgroud 
fg_bg_subctracor = cv2.createBackgroundSubtractorMOG2(50, 25, False)
#fg_bg_subctracor = cv2.createBackgroundSubtractorKNN(50, 200, False)

# Frame counting 
frameCount = 0

while True:
    _, frame = cap.read()
    frameCount += 1
    
    # Apply subtractor to frames from video capturing
    foreground_mask = fg_bg_subctracor.apply(frame)

	# Count all the white (changing) pixels within the mask
    count = np.count_nonzero(foreground_mask)
    print('Frame: %d, Pixel Count: %d' % (frameCount, count))

    # Convert colors to gray scale because of using face cascade 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Determine how many pixels do you want to detect to be considered as motion (changing pixels)
    if (frameCount > 20 and count > 550):
        cv2.putText(frame, 'Motion detected', (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Draw red rectangle on recognized moving face
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    else:
        # Draw green rectangle on recognized face
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', foreground_mask)

    # Quit program with escape key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()