import cv2
import matplotlib.pyplot as plt
import numpy as np

frame_counter = 0
kernel = np.ones((21,21),np.uint8)
cap = cv2.VideoCapture('./test3.mp4')
while(cap.isOpened()):
	ret, frame = cap.read()
	if (ret == True):
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		bars = frame_hsv[1020:1080, 665:1110]
		health_bar = bars[8:25, 17:432]
		mana_bar = bars[30:45, 17:430]
		## Thresholding with Saturation Thresh
		text_low = (0, 0, 180)
		text_high = (180, 60, 255)
		health_bar = cv2.inRange(health_bar, text_low, text_high)
		health_bar = health_bar[:, 150:270]
		helath_bar = cv2.resize(health_bar, None, fx = 7, fy = 7, interpolation = cv2.INTER_CUBIC)
		closing = cv2.morphologyEx(helath_bar, cv2.MORPH_CLOSE, kernel)
		#health_bar = cv2.resize()
		cv2.imshow('health', health_bar)
		#cv2.imshow('mana', mana_bar)
		if (frame_counter == 0):
			longshot = health_bar
		elif (frame_counter == 456):
			longshot = np.concatenate((longshot, health_bar), axis=0)
	else:

		'''
		plt.figure()
		plt.imshow(health_bar)
		plt.figure()
		plt.imshow(mana_bar)
		plt.show()
		'''
		cv2.imwrite('./health_test.png',longshot)
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
	   	break

	frame_counter += 1
cap.release()
cv2.destroyAllWindows() 