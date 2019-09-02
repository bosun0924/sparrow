import cv2
import matplotlib.pyplot as plt
import numpy as np

frame_counter = 0
kernel = np.ones((27,27),np.uint8)
cap = cv2.VideoCapture('./test2.mp4')
while(cap.isOpened()):
	ret, frame = cap.read()
	if (ret == True):
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		bars = frame_hsv[1020:1080, 665:1110]
		health_bar_o = bars[8:25, 17:432]
		mana_bar = bars[30:45, 17:430]
		## Thresholding with Saturation Thresh
		text_low = (60, 200, 100)
		text_high = (70, 255, 255)
		health_bar = cv2.inRange(health_bar_o, text_low, text_high)
		health_bar = cv2.blur(health_bar, (15,9))
		_, Thresh = cv2.threshold(health_bar,48,255,cv2.THRESH_BINARY)
		closing = cv2.morphologyEx(Thresh, cv2.MORPH_CLOSE, kernel)
		'''
		health_bar = health_bar[:, 150:270]
		helath_bar = cv2.resize(health_bar, None, fx = 7, fy = 7, interpolation = cv2.INTER_CUBIC)
		
		'''
		#health_bar = cv2.resize()
		cv2.imshow('health', closing)
		cv2.imshow('original', health_bar_o)
		#cv2.imshow('mana', mana_bar)
		if (frame_counter == 0):
			longshot = health_bar
		elif (frame_counter%25 == 0):
			longshot = np.concatenate((longshot, health_bar), axis=0)
	else:
		cv2.imwrite('./health_test.png',longshot)
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
	   	break
	frame_counter += 1
cap.release()
cv2.destroyAllWindows() 