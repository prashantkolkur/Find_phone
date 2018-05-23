
import pickle
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
import math

if __name__ == '__main__':
    path =  sys.argv[1]
    #print 'Loading the find_phone_model'
    with open('./find_phone_model.pk' ,'rb') as f:
    	find_phone_model = pickle.load(f)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    l=15
    B = []
    for i in range(0, img.shape[0], l):
        for j in range(0, img.shape[1], l):
           	A = img[i:i+l, j:j+l]
           	A = np.histogram(A, bins=256)
           	B.append(A[0])

    c = find_phone_model.predict(B)
        
    x_axis = []
    y_axis = []
    num_row_patch = math.ceil((img.shape[1]+0.0)/l)
    if np.where(c==1)[0].tolist():
        for i in np.where(c==1)[0].tolist():
            y_axis.append(int(i/num_row_patch)*l - l)
            y_axis.append(int(i/num_row_patch)*l + l)
            x_axis.append(int(i%num_row_patch)*l - l)
            x_axis.append(int(i%num_row_patch)*l + l)

        x_point = round(((max(x_axis)+min(x_axis)+0.0)/2)/490,2)
        y_point = round(((max(y_axis)+min(y_axis)+0.0)/2)/326,2)

        print x_point,y_point

        img[min(y_axis):max(y_axis),min(x_axis):max(x_axis)][0:2,:] = 255
        img[min(y_axis):max(y_axis),min(x_axis):max(x_axis)][-2:,:] = 255
        img[min(y_axis):max(y_axis),min(x_axis):max(x_axis)][:,0:2] = 255
        img[min(y_axis):max(y_axis),min(x_axis):max(x_axis)][:,-2:] = 255
        
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
    	print 0.50, 0.50

