import cv2
import numpy as np 
import time


class plate():
    def __init__(self, img, type_of_plate):
        self.type_of_plate = type_of_plate
        self.img = img
        if (self.type_of_plate == 'long_plate'):
            self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))
            self.type_of_plate = 0
        if (self.type_of_plate == 'square_plate'):
            self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(26, 7))
            self.type_of_plate = 1

    def ratioCheck(self, area, width, height):
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        if (self.type_of_plate == 0):
            aspect = 4.2727
            min = 30*aspect*30  # minimum area
            max = 125*aspect*125  # maximum area

            rmin = 3
            rmax = 7
        if (self.type_of_plate == 1):
            aspect = 1.4
            min = 100*aspect*100  # minimum area
            max = 175*aspect*175  # maximum area

            rmin = 1
            rmax = 3
        if (area < min or area > max) or (ratio < rmin or ratio > rmax):
            return False
        return True

    def validateRotationAndRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        if (width > height):
            angle = - rect_angle
        else:
            angle = 90 + rect_angle
        if angle>15:
            return False
        if height == 0 or width == 0:
            return False
        area = height*width
        if not self.ratioCheck(area,width,height):
            return False
        else:
            return True
        

    def process(self, img):
        self.blur = cv2.GaussianBlur(self.img, (7,7), 0)
        self.gray = cv2.cvtColor(self.blur, cv2.COLOR_BGR2GRAY)
        self.sobelx = cv2.Sobel(self.gray,cv2.CV_8U, 1, 0, ksize=3)
        __, threshold = cv2.threshold(self.sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #print(threshold)
        element = self.element_structure
        self.morph = threshold.copy()
        cv2.morphologyEx(src=threshold, op=cv2.MORPH_CLOSE, kernel=element, dst=self.morph)
        cv2.imshow('morph', self.morph)
        cv2.waitKey(0)
        return self.morph

    def cleanPlate(self,plate):
        self.img_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        __,self.img_thresh = cv2.threshold(self.img_gray,150, 255, cv2.THRESH_BINARY)
        new,self.contours, hierarchy = cv2.findContours(self.img_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.contours:
            areas = [cv2.contourArea(c) for c in self.contours]
            max_index = np.argmax(areas)

            max_cnt = self.contours[max_index]
            max_cntArea = areas[max_index]
            x,y,w,h = cv2.boundingRect(max_cnt)
            
            if not self.ratioCheck(max_cntArea,w,h):
                return plate,None

            self.cleaned_final = self.img_thresh[y:y+h, x:x+w]
            return self.cleaned_final,[x,y,w,h]

        else:
            return plate,None

    def extract_contours(self, morph):
        _, self.contours ,_ = cv2.findContours(self.morph, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        return self.contours

    def CleanAndRead(self, contours):
        for i,cnt in enumerate(self.contours):
            min_rect = cv2.minAreaRect(cnt)
            if self.validateRotationAndRatio(min_rect):
                x,y,w,h = cv2.boundingRect(cnt)
                self.plate_img = self.img[y:y+h,x:x+w]
                self.clean_plate, self.rect = self.cleanPlate(self.plate_img)
                if self.rect:
                    x1,y1,w1,h1 = self.rect
                    x,y,w,h = x+x1,y+y1,w1,h1
                    self.crop_contour = self.img[y:y+h,x:x+w]  
                    cv2.imshow("Cleaned Plate",self.crop_contour)
                    cv2.waitKey(0)
                    #cv2.imwrite("character2/character"+ str(time.time())+'.jpg',self.crop_contour)
                    return self.crop_contour
                else:
                    return None


    
