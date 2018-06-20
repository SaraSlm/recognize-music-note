import numpy as np
import cv2
import copy
from keras.models import load_model
import os
from keras.preprocessing import image
from pygame import mixer
from playsound import playsound

model = load_model('type2.h5')
color_model = load_model('color2.h5')
image2 = cv2.imread('mine.jpg')
cv2.imshow('orig',image2)
cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
note_arr=[]
note_arr_color=[]
n=0
for i, ctr in enumerate(sorted_ctrs):

    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi_first = image2[y:y+h, x:x+w]
    # show ROI
    # cv2.imshow('segment no:'+str(i),roi_first)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(roi_first, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    white = cv2.bitwise_not(thresh)

    horizontal = copy.deepcopy(thresh)
    horizontalsize = np.size(horizontal, 1) / 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontalsize),1))
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations = 1)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations = 1)
    horizontal = cv2.bitwise_not(horizontal)

    vertical = copy.deepcopy(thresh)
    verticalsize = np.size(vertical, 0) / 15
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, ( 1,int(verticalsize)))
    vertical = cv2.erode(vertical, verticalStructure, iterations = 1)
    vertical = cv2.dilate(vertical, verticalStructure, iterations = 1)
    vertical = cv2.bitwise_not(vertical)

    kernel_e = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(vertical, kernel_e, iterations = 1)
    kernel_o = np.ones((6, 6), np.uint8)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel_o)

    kernel_e = np.ones((2,2), np.uint8)
    final_erosion = cv2.erode(opening, kernel_e, iterations = 1)
    final_erosion = cv2.bitwise_not(final_erosion)
    im2, ctrs, hier = cv2.findContours(final_erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = vertical[y:y+h, x:x+w]
        row, col = roi.shape[:2]
        bottom = roi[row-2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        bordersize = y
        roi = cv2.copyMakeBorder(roi, top=bordersize, bottom=bordersize+h, left=x, right=x+w, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        height, width = horizontal.shape[:2]
        T = np.float32([[1, 0, 0], [0, 1,0]])
        roi = cv2.warpAffine(roi, T, (width, height))
        if( x - 25 > 0 ):
            left_padding=x-25
        else:
            left_padding=x
        cropped_roi=roi[:, left_padding:x+30]
        # cv2.imshow('c_r',cropped_roi)
        # cv2.waitKey(0)
        AND = cv2.bitwise_and(roi, horizontal)
        # cv2.imshow('AND',AND)
        # cv2.waitKey(0)

        # kernel_ee = np.ones((3,3), np.uint8)
        # erosion = cv2.erode(AND, kernel_ee, iterations = 1)
        # cv2.imshow('Erosion', erosion)
        # cv2.waitKey(0)
        if( x - 20 > 0 ):
            left_padding=x-25
        else:
            left_padding=x
        cropped = AND[:, left_padding:x+30]
        cv2.imwrite('./readed/segment_no'+str(n)+'.jpg', AND)
        n = n + 1
        if( cropped.shape[0] > 0 and cropped.shape[1] > 0 ):
            img_scaled = cv2.resize(cropped, (150, 150), interpolation = cv2.INTER_AREA)
            # cv2.imshow('img_scaled',img_scaled)
            # cv2.waitKey(0)
            backtorgb = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)
            # cv2.imshow('backtorgb',backtorgb)
            # cv2.waitKey(0)
            x = image.img_to_array(backtorgb)
            x /= 255.
            x = np.expand_dims(x, axis=0)

            # cv2.imshow('ver',roi_first)
            # cv2.waitKey(0)
            # cropped_roi = roi_first[:, left_padding:x+55]
            img_scaled_vertical = cv2.resize(cropped_roi, (150, 150), interpolation = cv2.INTER_AREA)
            # cv2.imshow('img_scaled_vertical',img_scaled_vertical)
            # cv2.waitKey(0)
            backtorgb_ver = cv2.cvtColor(img_scaled_vertical, cv2.COLOR_GRAY2RGB)
            # cv2.imshow('backtorgb_ver',backtorgb_ver)
            # cv2.waitKey(0)
            c = image.img_to_array(backtorgb_ver)
            c /= 255.
            c = np.expand_dims(c, axis=0)

            k_v_color = {0: 'black', 1: 'circle', 2: 'white'}
            k_v = {0: 'bmol', 1: 'diez',2: 'do_1', 3: 'do_2', 4: 'do_3', 5: 'fa_1', 6: 'fa_2', 7: 'la_1', 8: 'la_2', 9: 'la_3', 10: 'mi_1', 11: 'mi_2',
                 12: 'not_imp', 13: 're_1', 14: 're_2', 15: 'si_1', 16: 'si_2', 17: 'si_3', 18: 'sol_1', 19: 'sol_2', 20: 'sol_3', 21: 'sol_key'}

            if(max((model.predict(x))[0])>0.34):
                y = model.predict_classes(x)
                note_arr.append(k_v[y[0]])

                y_color=color_model.predict_classes(c)

                note_arr_color.append(k_v_color[y_color[0]])
print(note_arr)
print(note_arr_color)
for note in note_arr:
    for note_c in note_arr_color:
    # if(note!='bmol' and note!='diez' and note!='not_imp' and note!='sol_key'):
        if(note=='do_1'):
            if(note_c=='black'):
                playsound('./note_voice/do_1_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/do_1_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/do_1_white.mp3')
        elif(note=='do_2'):
            if(note_c=='black'):
                playsound('./note_voice/do_2_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/do_2_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/do_2_white.mp3')
        elif(note=='do_3'):
            if(note_c=='black'):
                playsound('./note_voice/do_3_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/do_3_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/do_3_white.mp3')
        elif(note=='fa_1'):
            if(note_c=='black'):
                playsound('./note_voice/fa_1_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/fa_1_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/fa_1_white.mp3')
        elif(note=='fa_2'):
            if(note_c=='black'):
                playsound('./note_voice/fa_2_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/fa_2_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/fa_2_white.mp3')
        elif(note=='la_1'):
            if(note_c=='black'):
                playsound('./note_voice/la_1_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/la_1_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/la_1_white.mp3')
        elif(note=='la_2'):
            if(note_c=='black'):
                playsound('./note_voice/la_2_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/la_2_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/la_2_white.mp3')
        elif(note=='la_3'):
            if(note_c=='black'):
                playsound('./note_voice/la_3_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/la_3_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/la_3_white.mp3')
        elif(note=='mi_1'):
            if(note_c=='black'):
                playsound('./note_voice/mi_1_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/mi_1_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/mi_1_white.mp3')
        elif(note=='mi_2'):
            if(note_c=='black'):
                playsound('./note_voice/mi_2_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/mi_2_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/mi_2_white.mp3')
        elif(note=='re_1'):
            if(note_c=='black'):
                playsound('./note_voice/re_1_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/re_1_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/re_1_white.mp3')
        elif(note=='re_2'):
            if(note_c=='black'):
                playsound('./note_voice/re_2_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/re_2_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/re_2_white.mp3')
        elif(note=='si_1'):
            if(note_c=='black'):
                playsound('./note_voice/si_1_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/si_1_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/si_1_white.mp3')
        elif(note=='si_2'):
            if(note_c=='black'):
                playsound('./note_voice/si_2_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/si_2_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/si_2_white.mp3')
        elif(note=='si_3'):
            if(note_c=='black'):
                playsound('./note_voice/si_3_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/si_3_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/si_3_white.mp3')
        elif(note=='sol_1'):
            if(note_c=='black'):
                playsound('./note_voice/sol_1_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/sol_1_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/sol_1_white.mp3')
        elif(note=='sol_2'):
            if(note_c=='black'):
                playsound('./note_voice/sol_2_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/sol_2_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/sol_2_white.mp3')
        elif(note=='sol_3'):
            if(note_c=='black'):
                playsound('./note_voice/sol_3_black.mp3')
            elif(note_c=='circle'):
                playsound('./note_voice/sol_3_circle.mp3')
            elif(note_c=='white'):
                playsound('./note_voice/sol_3_white.mp3')
