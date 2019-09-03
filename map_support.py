import numpy as np
import cv2

dark = 4
thr = 18
max_val = 255

def region_of_interest(image,corner = 'right'): 
    #get the resolution of the image
    height, width = image.shape
    map_perc = 0.85
    #set up the map extracting area
    map_height_limit = int(0.7*height)
    #set the cropping polygons
    if corner == 'right':
        map_width_limit = int(map_perc*width)
        area = [(map_width_limit, height),(map_width_limit, map_height_limit),(width, map_height_limit),(width, height),]
        crop_area = np.array([area], np.int32)
    if corner == 'left':
        map_width_limit_left = int((1-map_perc)*width)
        area = [(0, height),(0, map_height_limit),(map_width_limit_left, map_height_limit),(map_width_limit_left, height),]
        crop_area = np.array([area], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def get_map_boundaries_right(x1,x2,y1,y2,a,c,b,d):
    xmin=min(x1,x2)
    xmax=max(x1,x2)
    ymin=min(y1,y2)
    ymax=max(y1,y2)
    if (abs(x1-x2)<5):#verdical boudary
        if (xmax<a):
            a=xmin
        elif (xmin>c):
            c=xmax
    if (abs(y1-y2)<5):#horizontal boudary
        if (ymax<b):
            b=ymin
        elif (ymin>d):
            d=ymax
    return [a,c,b,d]

def get_map_boundaries_left(x1,x2,y1,y2,a,c,b,d):
    xmin=min(x1,x2)
    xmax=max(x1,x2)
    ymin=min(y1,y2)
    ymax=max(y1,y2)
    if (abs(x1-x2)<5):#verdical boudary
        if (xmin>a):
            a=xmax
        elif (xmax<c):
            c=xmin
    if (abs(y1-y2)<5):#horizontal boudary
        if (ymax<b):
            b=ymin
        elif (ymin>d):
            d=ymax
    return [a,c,b,d]

def finding_minimap(image, lines, corner = 'right'):
    #get the hight,lenth of the image.
    y, x, c = image.shape
    #initialize the boudary coordinates(outside of the image)
    a = int(0.96*x)
    c = int(0.96*x)
    a_left = int(0.04*x)
    c_left = int(0.04*x)
    b = int(y*0.96)
    d = int(y*0.96)
    map = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines in the mini map
            ##right corner
            if corner == 'right':
                (a,c,b,d) = get_map_boundaries_right(x1,x2,y1,y2,a,c,b,d)
            ##left coner
            if corner == 'left':
                (a_left,c_left,b,d) = get_map_boundaries_left(x1,x2,y1,y2,a_left,c_left,b,d)

    if corner == 'right':
        map_co = [a,b,c,d]
    if corner == 'left':
        map_co = [a_left,b,c_left,d]
    return map_co

def display_map(image,a,b,c,d):
    map = np.zeros_like(image)
    #horizontal
    cv2.line(map, (a, b), (c, b), (0, 255, 0), 3)
    cv2.line(map, (a, d), (c, d), (0, 255, 0), 3)
    #verdical
    cv2.line(map, (a, b), (a, d), (0, 255, 0), 3)
    cv2.line(map, (c, b), (c, d), (0, 255, 0), 3)
    return map

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return line_image

def capture_map(cap, corner = 'right'):
    ret, frame = cap.read()
    if (ret == True):
        frame = cv2.resize(frame, (1920, 1080))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #Thresholding to get map edges highlighted
        ret, o6 = cv2.threshold(gray_image, dark, thr, cv2.THRESH_BINARY_INV)
        map_region = region_of_interest(o6, corner)
        #set up hough transformation
        rho = 2
        theta = np.pi/180
        threshold = 100
        lines = cv2.HoughLinesP(map_region,rho, theta, threshold, np.array ([]), minLineLength=45, maxLineGap=8)
        map_info = finding_minimap(frame, lines, corner)
        return map_info
    else:
        return [0,0,0,0]

def initial_detecting(cap,dark,thr,max_val,map_corner = 'right',foundMap = False,map_coord_stack = np.empty((21,4), dtype = int), k = 0):
    print("##############initializing#############")
    while (foundMap==False):
        for i in range(21):
            map_coord_stack[i] = capture_map(cap,map_corner)
        #getting the initial map coordinates
        a = np.argmax(np.bincount(map_coord_stack[:,0]))
        b = np.argmax(np.bincount(map_coord_stack[:,1]))
        c = np.argmax(np.bincount(map_coord_stack[:,2]))
        d = np.argmax(np.bincount(map_coord_stack[:,3]))
        minimap = miniMap(a,b,c,d)
        box_centre = minimap.get_centre()
        print('Minimap Centre Location: {0}'.format(box_centre))
        if (map_corner=='right')and(box_centre[0]>1770)and(box_centre[0]<1830)and (box_centre[1]>900):
            foudnMap = True
            break
        elif (map_corner=='left')and(box_centre[0]>105)and(box_centre[0]<150)and (box_centre[1]>900):
            foudnMap = True
            break
        else:
            map_corner = 'left' if (map_corner == 'right') else 'right'
            map_coord_stack = np.empty((21,4), dtype = int)
    return [a,b,c,d,map_corner]
'''_______________________________________________________'''

class miniMap():
    def __init__(self,a,b,c,d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def get_centre(self):
        return [int((self.a+self.c)/2),int((self.b+self.d)/2)]

    def get_boudaries(sefl):
        return [self.a, self.b, self.c, self.d]