import numpy as np
import cv2
import copy
from keras.models import load_model
import os
from keras.preprocessing import image
from pygame import mixer
from playsound import playsound

image_dir_path = 'C:/Users/Salimian/Desktop/vision/notes_to_music/vision_project/dataset/validation/color'
image_dir_path_to = 'C:/Users/Salimian/Desktop/vision/notes_to_music/vision_project/dataset2/validation/color'
for dirs in os.walk(image_dir_path):
    for d in dirs[1]:
        for files in os.walk(image_dir_path+'/'+d):
            for filename in files[2]:
                # print(image_dir_path+'/'+d+'/'+filename)
                roi = cv2.imread(image_dir_path+'/'+d+'/'+filename)
                gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)


                #binary image
                ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
                white = cv2.bitwise_not(thresh)

                #horizontal image including LINES
                horizontal = copy.deepcopy(thresh)
                # print(np.size(horizontal,1))
                horizontalsize = np.size(horizontal,1) / 30
                horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontalsize),1))
                horizontal = cv2.erode(horizontal, horizontalStructure, iterations = 1)
                horizontal = cv2.dilate(horizontal, horizontalStructure, iterations = 1)
                cv2.imshow('horizontal', horizontal)
                horizontal = cv2.bitwise_not(horizontal)
                cv2.waitKey(0)
                # kernel = np.ones((5,5), np.uint8)
                # closing = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel)
                # cv2.imshow('Closing', horizontal)
                # cv2.waitKey(0)
                #verticall image including NOTES
                vertical = copy.deepcopy(thresh)
                verticalsize = np.size(vertical,0) / 30
                verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, ( 1,int(verticalsize)))
                vertical = cv2.erode(vertical, verticalStructure, iterations = 1)
                vertical = cv2.dilate(vertical, verticalStructure, iterations = 1)
                vertical = cv2.bitwise_not(vertical)
                # cv2.imshow("vertical", vertical)
                # cv2.waitKey(0)

                kernel_e = np.ones((3,3), np.uint8)
                erosion = cv2.erode(vertical, kernel_e, iterations = 1)
                # cv2.imshow('erosion',erosion)
                # cv2.waitKey(0)

                kernel_o = np.ones((6,6), np.uint8)
                opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel_o)
                # cv2.imshow('Opening', opening)
                # cv2.waitKey(0)

                canny = cv2.Canny(opening, 60, 120)
                canny = cv2.bitwise_not(canny)
                # cv2.imshow('Canny', canny)
                # cv2.waitKey(0)

                kernel_e = np.ones((2,2), np.uint8)
                final_erosion = cv2.erode(canny, kernel_e, iterations = 1)
                # cv2.imshow('final erosion',final_erosion)
                # cv2.waitKey(0)


                #find contours
                final_erosion = cv2.bitwise_not(final_erosion)
                # cv2.imshow('final_erosion',final_erosion)
                # cv2.waitKey(0)
                roi2 = vertical[:, :]
                row, col = roi2.shape[:2]
                bottom = roi2[row-2:row, 0:col]
                mean = cv2.mean(bottom)[0]

                ret, roi2 = cv2.threshold(roi2, 127, 255, cv2.THRESH_BINARY)
                height, width = horizontal.shape[:2]
                T = np.float32([[1, 0, 0], [0, 1,0]])
                roi2 = cv2.warpAffine(roi2, T, (width, height))
                AND = cv2.bitwise_and(roi2, horizontal)
                # cv2.imshow('AND',AND)
                # cv2.waitKey(0)
                # kernel_ee = np.ones((3,3), np.uint8)
                # erosion = cv2.erode(AND, kernel_ee, iterations = 1)
                # cv2.imshow('Erosion', erosion)
                # cv2.waitKey(0)



        # cv2.imshow('cr',cropped)
        # cv2.waitKey(0)
        # kernel = np.ones((5,5), np.uint8)

        # Now we erode
        # cropped = cv2.erode(cropped, kernel, iterations = 1)
        # cv2.imshow('Erosion', cropped)
        # cv2.waitKey(0)
        #
        # # #
        # cropped = cv2.dilate(cropped, kernel, iterations = 1)
        # cv2.imshow('Dilation', cropped)
        # cv2.waitKey(0)
        # #
        # # Opening - Good for removing noise
        # cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('Opening', cropped)
        # cv2.waitKey(0)
        # #
        # # # Closing - Good for removing noise
        # cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)

        # cropped = cv2.Canny(cropped, 60, 120)
        # cropped = cv2.erode(cropped, kernel_e, iterations = 1)
        # cropped = cv2.bitwise_not(cropped)
        # cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)
        # cropped = cv2.dilate(cropped, kernel, iterations=1)
        # cropped = cv2.dilate(cropped, kernel, iterations=1)
        # cv2.imshow('Closing', cropped)
        # cv2.waitKey(0)
                print(image_dir_path_to+'/'+d+'/'+filename)
                cv2.imwrite(image_dir_path_to+'/'+d+'/'+filename,horizontal)
    break
