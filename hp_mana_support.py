import cv2
import numpy as np

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return line_image

def get_bar(lines):
    a = 1920
    b = 1080
    c = 0
    d = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines of the health bar
            if (abs(y1-y2)<=2):#horizontal boudary
                a = min(x1,x2) if (a > min(x1,x2)) else a
                c = max(x1,x2) if (c < max(x1,x2)) else c
                b = y1 if (b > y1) else b
                d = y1 if (d < y1) else d
        return [a,b,c,d]
    else:
        return None

def extracting_health(image_RGB, bottom_region):
    #setting green health bar binning parameters
    low_green = np.array([60,230,0])
    high_green = np.array([66,255,255])
    #color select the green health bar
    health_bar = cv2.inRange(bottom_region, low_green, high_green)
    #hough transformation
    rho = 2
    theta = np.pi/180
    threshold = 100
    lines = cv2.HoughLinesP(health_bar,rho, theta, threshold, np.array ([]), minLineLength=60, maxLineGap=10)
    if lines is not None:
        return get_bar(lines)
    else:
        return None

def extracting_mana(image_RGB, bottom_region):
    #setting green health bar binning parameters
    low_blue = np.array([100,200,0])
    high_blue = np.array([106,255,255])
    #color select the green health bar
    mana_bar = cv2.inRange(bottom_region, low_blue, high_blue)
    #hough transformation
    rho = 2
    theta = np.pi/180
    threshold = 100
    lines = cv2.HoughLinesP(mana_bar,rho, theta, threshold, np.array ([]), minLineLength=60, maxLineGap=10)
    if lines is not None:
        return get_bar(lines)
    else:
        return None

def health_bar_perc(health_bar):
    kernel = np.ones((27,27),np.uint8)
    bar_low = (60, 200, 100)
    bar_high = (70, 255, 255)
    health_bar = cv2.inRange(health_bar, bar_low, bar_high)
    health_bar = cv2.blur(health_bar, (15,9))
    _, Thresh = cv2.threshold(health_bar,48,255,cv2.THRESH_BINARY)
    closing = cv2.morphologyEx(Thresh, cv2.MORPH_CLOSE, kernel)
    health_perc = round((cv2.mean(closing)[0])*100/255)
    return health_perc

def mana_bar_perc(mana_bar):
    kernel = np.ones((27,27),np.uint8)
    bar_low = (100, 200, 100)
    bar_high = (106, 255, 255)
    mana_bar = cv2.inRange(mana_bar, bar_low, bar_high)
    mana_bar = cv2.blur(mana_bar, (15,9))
    _, Thresh = cv2.threshold(mana_bar,48,255,cv2.THRESH_BINARY)
    closing = cv2.morphologyEx(Thresh, cv2.MORPH_CLOSE, kernel)
    mana_perc = round((cv2.mean(closing)[0])*100/255)
    return mana_perc