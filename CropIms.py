
import cv2
import numpy as np
import os
import csv

'''
Function for cropping images based on irregualrities in contour
First of all, binarize image
Extract contours afterwards
Iterate over all contours
'''
def crop_image(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    threshold = 1000
    min_contour_size = 0
    w = 500
    h = 500
    for contour in contours:
        #Check if contour is acually contour with expansion
        if contour.shape[0] > 1:
            #Exclude contours of unsifficient soze
            if cv2.contourArea(contour) < min_contour_size:
                continue
            #Exclude contours at the edges of image
            x, y, w, h = cv2.boundingRect(contour)
            if x < 2 or y < 2 or (x+w) > (image.shape[1]-2) or (y+h) > (image.shape[0]-2):
                continue
            if contour.ndim == 2:
                points= contour.squeeze()
            elif contour.ndim == 3:
                points = contour.squeeze(1)
            else:
                points = contour
              
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if(x<128):
                x_min = 0
                x_max = 256
            elif(x > 1920):
                x_max = 2048
                x_min = 1792
            else:
                x_min = x -128
                x_max = x +128
            if(y<128):
                y_min = 0
                y_max = 256
            elif(y > 896):
                y_max = 1024
                y_min = 768
            else:
                y_min = y -128
                y_max = y +128
            cropped_image = image[y_min:y_max, x_min:x_max, 0]
            return(cropped_image)

'''
Get image paths
'''
with open('image_paths.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
im_paths  = np.array(data)
im_paths = im_paths[1:, 1]

'''
Open all images and crop 
If label is NOK crop with function crop_image
Otherwise crop randomly
'''
for path in im_paths: 
    im = cv2.imread('path_to_ims.tiff'))
    path = path.split('/')[-1]
    if(path.split('_')[-1] == 'NOK.tiff'):
        cropped = crop_image(im)
        if(cropped is not None):
            cv2.imwrite('Cropped_Ims/' + path, cropped)
            print('Saved cropped im (NOK)')
    else:
        cropped = im[400:656, 500:756, 0]
        cv2.imwrite('Cropped_Ims/' + path, cropped)
        print('Saved cropped im (OK)')
    
