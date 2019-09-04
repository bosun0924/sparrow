## Sparrow.GG 2019
## League of Legends Computer Vision Analysis

import cv2
import numpy as np
import math
from map_support import *
from digitize import *
from Regression import *
from hp_mana_support import *
	
#############################################
####          Initialization             ####
cap = cv2.VideoCapture('./test1.mp4')
map_corner = 'right'
res = (1920, 1080)
frame_cntr = 0
# kernel for some morphological operations
kernel = np.ones((5,5),np.uint8)
with open("Output.txt", "w") as text_file:
	print("________Gaming Data_______", file=text_file)

#############################################
#### K-NN Models for Money and Player Level Digit Recog ####
npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
kNearest1 = cv2.ml.KNearest_create()                   # instantiate KNN object
kNearest1.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

#############################################
#### K-NN Models for HP and Mana Recog ####
npaClassifications_hm = np.loadtxt("classifications_hm.txt", np.float32)                  # read in training classifications
npaFlattenedImages_hm = np.loadtxt("flattened_images_hm.txt", np.float32)                 # read in training images
npaClassifications_hm = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
kNearest2 = cv2.ml.KNearest_create()                   # instantiate KNN object
kNearest2.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

#############################################
####          Locate the minimap         ####
(a,b,c,d,map_corner) = initial_detecting(cap,dark,thr,max_val)
#### Define the codec and create VideoWriter object ####
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('minimap.avi',fourcc, 30.0, (252,252))

#############################################
####          Skills Initiation          ####

# skills, cooling or ready to use, level
class skill:
    def __init__(self, loc, name, upgradability):
        self.loc = loc
        self.name = name
        self.img = frameThresh_skills
        self.upgradability = upgradability
    def get_state(self):
        brightness = cv2.mean(self.img[self.loc[0]:self.loc[1],self.loc[2]:self.loc[3]])
        #if the skill level is not 0, which means not developed
        #Determine whether it is cooling or ready
        if (self.get_skill_level() != 0):
            return 'Cooling' if (brightness[0] < 50) else 'Ready'
			#return 'Y' if (brightness[0] < 50) else 'N'
        else:
            #otherwise, it is not developed
            return 'NA'
    def get_name(self):
        return self.name
    def get_img(self):
        return img[self.loc[0]:self.loc[1],self.loc[2]:self.loc[3]]
    def get_skill_level(self):
        if (self.upgradability == True):
            #getting the brightness/luminus of the skill dot area
            luminus = cv2.sumElems(self.img[1012:1019,self.loc[2]:self.loc[3]])
            #One skill dot worths about 9754 luminus
            skill_level = round(luminus[0]/9000)
            return skill_level
        else:
            return None

ret, frame = cap.read()
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
low_limit = np.array([0, 0, 128])# Boarderline for bright and dark
high_limit = np.array([180, 255, 255])
frameThresh_skills = cv2.inRange(frame_hsv, low_limit, high_limit)
## Create skill objects
skill_0 = skill([950,  990,  680,  722], '0', False)
skill_Q = skill([950, 1005,  725,  787], 'Q', True)
skill_W = skill([950, 1005,  798,  853], 'W', True)
skill_E = skill([950, 1005,  861,  921], 'E', True)
skill_R = skill([950, 1005,  930,  990], 'R', True)
skill_D = skill([950,  990, 1005, 1048], 'D', False)
skill_F = skill([950,  990, 1055, 1100], 'F', False)
skills = [ skill_0, skill_Q, skill_W, skill_E, skill_R, skill_D, skill_F]

#############################################
####       HP MANA Bars Initiation       ####
#spotting on the bottom region
hp_mana_bars_region = frame_hsv[1020:1080, 665:1110]
h_co = extracting_health(frame, hp_mana_bars_region)
#h_co = [17,8,430,25] # When the recording 1080p
m_co = extracting_mana(frame, hp_mana_bars_region)
#m_co = [17,30,430,45] # When the recording 1080p
#############################################
####              Main Loop              ####
print("__Enter main loop__")
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		# 0. Prepair
		frame_cntr += 1
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		######################################################################
		#####################################################################
		# 1. Minimap
		map_info = display_map(frame, a,b,c,d)
        #Show the whole image with mini map in a box at the corner
		result_image = cv2.addWeighted(frame, 0.8, map_info, 1, 1)
		cv2.imshow("Image", cv2.resize(result_image, (800,450)))
        #minimap cropping box (a,b,c,d)
		if (a==c)or(b==d):
			result_image_map = np.zeros((252,252))
		else:
			ver_l = min(a,c)
			ver_r = max(a,c)
			result_image_map = cv2.resize(frame[b:d,ver_l:ver_r], (252,252))
		cv2.imshow("mini map", result_image_map)
		out.write(result_image_map)
		frame_map = miniMap(a,b,c,d)
		box_centre = frame_map.get_centre()

        #######################################################
        # Check if the map resized or moved to the other corner

		(a1,b1,c1,d1) = capture_map(cap,map_corner)
		new_centre = [int((a1+c1)/2),int((b1+d1)/2)]
		#print(a1,b1,c1,d1)
		#print('Minimap Centre Location: {}'.format((a,b,c,d)))
		dx = abs(box_centre[0]-new_centre[0])
		dy = abs(box_centre[1]-new_centre[1])
		if (new_centre != [0,0]):
			if (map_corner=='right'):
				if (new_centre[0]>=1770)and(new_centre[0]<=1830)and(box_centre[1]>=900):
                #image stabilization
                #if the new centre is significantly away from the old one(2 pixels horizontal and verdical)
					if ((dx > 4) and (dy > 4)):
						if abs(dx-dy)<=6 :
							[a,b,c,d]=[a1,b1,c1,d1]
				else:
					map_corner = 'left'
					print('Minimap changed to left')
					(a,b,c,d) = capture_map(cap,map_corner)

			elif (map_corner=='left'):
				if (new_centre[0]>=102)and(new_centre[0]<=147)and (new_centre[1]>=900):
                    #image stabilization
                    #if the new centre is significantly away from the old one(2 pixels horizontal and verdical)
					if ((dx > 4) and (dy > 4)):
						if abs(dx-dy)<=6 :
							[a,b,c,d]=[a1,b1,c1,d1]
				else:
					map_corner = 'right'
					print('Minimap changed to right')
					(a,b,c,d) = capture_map(cap,map_corner) 
		else: break
		######################################################################
		#####################################################################
		# 2. Allies Level
		##############_______________Allies_______________################
		# The Allies' heads move with the minimap
		# The locations of their heads have a linear relationship with the 
		# minimap's location (a,b,c,d), and the minimap size percentage

		mini_map = [a,b,c,d]
		if (mini_map[0]>960):# if map on the right
			allies_frame = frame_hsv[int(mini_map[1]- (mini_map[2]-mini_map[0])*0.38):mini_map[1],mini_map[0]:1920]
			# Mini map percentage
			miniMap_size = (((mini_map[3]-mini_map[1])-190)/62)*100
			al_levels = mini_map
			al_levels.append(miniMap_size)
			# Regression Returns Allies Levels
			allies_coe = al_level_coe(al_levels)
			#print(allies_coe)
			intervel = allies_coe[2]
		else:# if map on the left
			allies_frame = frame_hsv[int(mini_map[1]- (mini_map[0]-mini_map[2])*0.38):mini_map[1],0:mini_map[0]]
			# Mini map percentage
			miniMap_size = (((mini_map[3]-mini_map[1])-190)/62)*100
			al_levels = mini_map
			al_levels.append(miniMap_size)
			# Regression Returns Allies Levels
			allies_coe = al_level_coe_left(al_levels)
			#print(allies_coe)
			intervel = allies_coe[2]
		# 
		allies_levels = []
		allies_levels_str = []
		text_low = (0, 0, 180)
		text_high = (180, 60, 255)
		for i in range(4):
			ally = allies_frame[int(allies_coe[1]-7):int(allies_coe[1]+7), int(allies_coe[0]-8+(i*intervel)):int(allies_coe[0]+10+(i*intervel))]
			ally = cv2.inRange(ally, text_low, text_high)
			ally = cv2.resize(ally, None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
			ally = cv2.morphologyEx(ally, cv2.MORPH_OPEN, kernel)
			text = get_text(ally, kNearest2, MIN_CONTOUR_AREA = 125)
			print('Ally {0} : {1} '.format(i+1, text), end = ',')
			allies_levels_str.append(text)
			allies_levels.append(ally)
		print('')
		''' Shows the levels of your allies
		cv2.imshow('Ally 1', allies_levels[0])
		cv2.imshow('Ally 2', allies_levels[1])
		cv2.imshow('Ally 3', allies_levels[2])
		cv2.imshow('Ally 4', allies_levels[3])
		'''
		######################################################################
		#####################################################################
		# 3. Skills States and Levels
		##############_______________Skills_______________###############
		low_limit = np.array([0, 0, 128])# Boarderline for bright and dark
		high_limit = np.array([180, 255, 255])
		frameThresh_skills = cv2.inRange(frame_hsv, low_limit, high_limit)
		## Create skill objects
		for skill in skills :
			skill.img = frameThresh_skills # Refresh the image 
			print('Skill', end=" ")
			print(skill.get_name(), end = " : ")
			print(skill.get_state(), end = ", Level: ")
			print(str(skill.get_skill_level()))

		######################################################################
		#####################################################################
		
		# 4. HP and Mana Bars
		hp_mana_bars_frame = frame_hsv[1020:1080, 665:1110]
		#print('Health Bar: {0}'.format(h_co))
		#print('Mana Bar: {0}'.format(m_co))
		health_bar = hp_mana_bars_frame[h_co[1]:h_co[3], h_co[0]:h_co[2]]
		print("Health Percentage: {0}%".format(health_bar_perc(health_bar)))
		mana_bar = hp_mana_bars_frame[m_co[1]:m_co[3], m_co[0]:m_co[2]]
		print("Mana Percentage: {0}%".format(mana_bar_perc(mana_bar)))
		'''
		health_bar = cv2.inRange(health_bar, text_low, text_high)
		health_bar = health_bar[:, 150:270]
		helath_bar = cv2.resize(health_bar, None, fx = 7, fy = 7, interpolation = cv2.INTER_CUBIC)
		print('Helath: {0}'.format(get_text(helath_bar, kNearest2, 150)))
		cv2.imshow('Health Bar', health_bar)
		'''

		######################################################################
		#####################################################################
		# 5. ExpBar and Player's Level

		##############_______________Exp Progress_______________###############
		exp_ROI_HSV = frame_hsv[940:1043, 624:667]
	    # Extract the Pruper bar
		dark_purple = (130, 160, 0)
		light_purple = (140, 255, 255)
		
		exp_extracting = cv2.inRange(exp_ROI_HSV, dark_purple, light_purple)
	    # Closing operation to get rid of Noise
		exp_extracting = cv2.morphologyEx(exp_extracting, cv2.MORPH_OPEN, kernel)
	    # Get the sum of all the lit pixels(255 each)
		lum, _, _, _ = cv2.sumElems(exp_extracting)
	    # Calibration #
	    # About every 8 lit pixels makes 1% of EXP to the next level
		progress = lum/(255*8)
		progress = round(progress/5)
		print('Exp Perct: {0}/20'.format(progress), end = "   ")

	    ##############_______________Exp Level_______________################
		level_ROI = frame_gray[1043:1060, 620:640]
		_, level_ROI = cv2.threshold(level_ROI, 127, 255, cv2.THRESH_BINARY)
		exp_level = cv2.resize(level_ROI, None, fx = 2, fy = 2)
		level = get_text(exp_level, kNearest1, 125)
		print('Current Level: {0}'.format(level))
		
		#####################################################################
		#####################################################################
		# 6. Money

		##############_______________Money_______________################
		money_region_gray = cv2.resize(frame_gray[1045:1070, 1200:1280], (160, 50))
		_, money = cv2.threshold(money_region_gray, 
                                                100, #threshold = 200
                                                255, #max value in image
                                                cv2.THRESH_BINARY)#binary thresh
		# Morphological operations
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		erosion = cv2.erode(money,kernel)
		closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
		# Textualization
		#cv2.imshow('money', closing)
		money = get_text(closing, kNearest1, 125)
		print('Money: {0}'.format(money))
		#####################################################################
		#####################################################################
		#####################################################################
		#####################################################################
		###########################  Writing  ###############################
		if (frame_cntr%(10) == 0): # Every 10 frames
			time_stamp = round((cap.get(cv2.CAP_PROP_POS_MSEC))/1000, 1)
			with open("Output.txt", "a") as text_file:
				print('Frame: {0}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)), file=text_file)
				print('Health Percentage: {0}%'.format(health_bar_perc(health_bar)), file=text_file)
				print('Mana Percentage: {0}%'.format(mana_bar_perc(mana_bar)), file=text_file)
				print('Money: {0}'.format(money), file=text_file)
				print('Exp Perct: {0}/20'.format(progress), end = "   ", file=text_file)
				print('Current Level: {0}'.format(level), file=text_file)
				for skill in skills :
					print('Skill', end=" ", file=text_file)
					print(skill.get_name(), end = " : ", file=text_file)
					print(skill.get_state(), end = ", Level: ", file=text_file)
					print(str(skill.get_skill_level()), file=text_file)
				for i in range(4):
					print('Ally{0} : Level {1} '.format(i+1, allies_levels_str[i]), end = '// ', file=text_file)
				print('', file=text_file)
				print('_______________________________________', file=text_file)	
	else:
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
	   	break
cap.release()
cv2.destroyAllWindows() 