import cv2
import numpy as np 
import time
import imutils
from plate_detection import plate
from matplotlib import pyplot as plt
from classCNN import NeuralNetwork


class character():
    def __init__(self, crop_contour, type_of_plate):
        self.type_of_plate = type_of_plate
        self.crop_contour = crop_contour
        if (self.type_of_plate == 'long_plate'):
            self.type_of_plate = 0
        if (self.type_of_plate == 'square_plate'):
            self.type_of_plate = 1

    def sort_contours(self,new_contours):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in self.new_contours]
        (self.new_contours, boundingBoxes) = zip(*sorted(zip(self.new_contours, boundingBoxes),
            key=lambda b:b[1][i], reverse=False))
        return self.new_contours

    def find_character(self, crop_contour):
        if (self.type_of_plate == 0):
            self.img_resize = imutils.resize(self.crop_contour,width=470)
            gray = cv2.cvtColor(self.img_resize, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 315, 0)
            return thresh
        if (self.type_of_plate == 1):
            self.img_resize = imutils.resize(self.crop_contour,width=280)
            plate_upper = self.img_resize[0:self.img_resize.shape[0]/2, 0:self.img_resize.shape[1]]
            plate_lower = self.img_resize[self.img_resize.shape[0]/2: self.img_resize.shape[0], 0:self.img_resize.shape[1]]
            if (plate_upper and plate_lower):
                charactersFound = plate_upper + plate_lower
                return charactersFound
            gray = cv2.cvtColor(charactersFound, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 315, 0)
            return thresh
    
    def extract_contours_character(self, thresh):
        _, self.contours ,_ = cv2.findContours(thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
        self.new_contours= sorted(self.contours, key = cv2.contourArea, reverse = True)[:9]
        self.new_contours= self.sort_contours(self.new_contours)
        return self.new_contours

    def convex_hull(self, thresh, new_contours):
        hull = []
        for i in range(len(self.new_contours)):
            hull.append(cv2.convexHull(self.new_contours[i], False))
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
                    
        for i in range(len(self.new_contours)):
            color = (255, 255, 255)
            self.convex = cv2.drawContours(drawing, hull, -1, color, -1)
            gray = cv2.cvtColor(self.convex, cv2.COLOR_BGR2GRAY)
            __, self.threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
            #print(self.threshold)
        # cv2.imshow('a',self.threshold)
        # cv2.waitKey(0)
        

    def show_contour(self, threshold):
        _, self.final_contours,_ = cv2.findContours(self.threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        self.new_final_contours= sorted(self.final_contours, key = cv2.contourArea, reverse = True)[:10]
        self.new_final_contours= self.sort_contours(self.new_final_contours)
        return self.new_final_contours

    def Read(self, new_final_contours):
        myNetwork = NeuralNetwork(modelFile="model/retrained_graph.pb",labelFile="model/retrained_labels.txt")
        self.Character = []
        List = []
        plate = ""
        for c in self.new_final_contours:
            x,y,w,h = cv2.boundingRect(c)
            self.Character.append(c)
            #print(w/h)
            if w/h < 1:
                start_row,start_col = int(y),int(x)-5
                end_row,end_col = int(y)+int(h)+5,int(x)+int(w)+5
                self.img_con = self.img_resize[start_row:end_row,start_col:end_col]
                # cv2.imshow('im', self.img_con)
                # cv2.waitKey(0)
                # cv2.imwrite("file_1/file"+ str(time.time())+'.jpg',self.img_con)
                # print(self.Character)
                self.lenList = len(self.Character)
                tensor = myNetwork.read_tensor_from_image(self.img_con,224)
                label = myNetwork.label_image(tensor)
                #print(label)
                List.append(label)
                #print(self.lenList)
                plate = plate + str(List[-1])
        print(plate)

    # def plot(self,convex):
    #     img_row_sum = np.sum(self.threshold, axis=1).tolist()
    #     _y = np.arange(self.threshold.shape[0])
    #     plt.plot(img_row_sum, _y)
    #     plt.show()
    #     img_col_sum = np.sum(self.threshold,axis=0).tolist()
    #     _x = np.arange(self.threshold.shape[1])
    #     plt.plot(_x, img_col_sum)
    #     plt.show()


    # def CleanAndRead(self, new_final_contours):
    #     if(self.type_of_plate == 0):
    #         self.Character = []
    #         for c in self.new_final_contours:
    #             x,y,w,h = cv2.boundingRect(c)
    #             self.Character.append(c)
    #             start_row,start_col = int(y),int(x)-5
    #             end_row,end_col = int(y)+int(h)+5,int(x)+int(w)+5
    #             self.img_con = self.img_resize[start_row:end_row,start_col:end_col]
    #             # cv2.imshow('im', self.img_con)
    #             # cv2.waitKey(0)
    #         self.lenList = len(Character)
    #     if(self.type_of_plate == 1):
    #         self.Character = []
    #         for c in self.new_final_contours:
    #             x,y,w,h = cv2.boundingRect(c)
    #             self.Character.append(c)
    #             if plate_upper:
    #                 start_row,start_col = int(y),int(x)-5
    #                 end_row,end_col = int(y)+int(h)+5,int(x)+int(w)+5
    #                 self.img_con = plate_upper[start_row:end_row,start_col:end_col]
    #                 # cv2.imshow('im', self.img_con)
    #                 # cv2.waitKey(0)
    #             if plate_lower:
    #                 start_row,start_col = int(y),int(x)-5
    #                 end_row,end_col = int(y)+int(h)+5,int(x)+int(w)+5
    #                 self.img_con = plate_lower[start_row:end_row,start_col:end_col]
    #                 # cv2.imshow('im', self.img_con)
    #                 # cv2.waitKey(0)
    #             self.lenList = len(self.Character)
                    

    def filter_img(self, new_final_contours):
        myNetwork = NeuralNetwork(modelFile="model/retrained_graph.pb",labelFile="model/retrained_labels.txt")
        for c in self.new_final_contours:
            x,y,w,h = cv2.boundingRect(c)
            if w/h < 1:
                start_row,start_col = int(y),int(x)-5
                end_row,end_col = int(y)+int(h)+5,int(x)+int(w)+5
                self.img_con = self.img_resize[start_row:end_row,start_col:end_col]
                tensor = myNetwork.read_tensor_from_image(self.img_con,224)
                label = myNetwork.label_image(tensor)
                cv2.imwrite("file_1/"+str(label)+"/"+ str(time.time())+'.jpg',self.img_con)
