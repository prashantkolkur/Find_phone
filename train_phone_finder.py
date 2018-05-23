
import pickle
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
import math

def train_phone(path):
    labels = pd.read_csv(path+'/labels.txt', delimiter=' ', header=None, names=['image', 'x', 'y'])
    labels['x'] = labels['x']*490
    labels['y'] = labels['y']*326
    labels = labels.set_index('image')
    
    train_files = labels.index
    l=15
    B = []
    label = []
    for file in train_files:
        img = cv2.imread(path+'/'+file, cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        pixel_x = int(labels.loc[file][0])
        pixel_y = int(labels.loc[file][1])
        x = [i for i in range(pixel_x-l, pixel_x+l, 1)]
        y = [i for i in range(pixel_y-l, pixel_y+l, 1)]
        for i in range(0, img.shape[0], l):
            for j in range(0, img.shape[1], l):
                A = img[i:i+l, j:j+l]
                A = np.histogram(A, bins=256)
                B.append(A[0])
                if ((j in x) and (i in y)):
                    label.append(1)
                else:
                    label.append(0)

    find_phone_model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    find_phone_model.fit(B,label)
    return find_phone_model

def train_accuracy(path,find_phone_model):
    labels = pd.read_csv(path+'/labels.txt', delimiter=' ', header=None, names=['image', 'x', 'y'])
    labels['x'] = labels['x']*490
    labels['y'] = labels['y']*326
    labels = labels.set_index('image')
    
    train_files = labels.index
    
    accuracy = []
    for file in train_files:
        label = []
        l = 15
        B = []
        img = cv2.imread(path+'/'+file, cv2.IMREAD_GRAYSCALE)
        #img = cv2.GaussianBlur(img,(5,5),0)
        pixel_x = int(labels.loc[file][0])
        pixel_y = int(labels.loc[file][1])
        x = [i for i in range(pixel_x-l, pixel_x+l, 1)]
        y = [i for i in range(pixel_y-l, pixel_y+l, 1)]
        for i in range(0, img.shape[0], l):
            for j in range(0, img.shape[1], l):
                A = img[i:i+l, j:j+l]
                A = np.histogram(A, bins=256)
                B.append(A[0])
                if ((j in x) and (i in y)):
                    label.append(1)
                else:
                    label.append(0)

        c = find_phone_model.predict(B)

        x_axis = []
        y_axis = []
        num_row_patch = math.ceil((img.shape[1]+0.0)/l)
        if np.where(c==1)[0].tolist():
            for i in np.where(c==1)[0].tolist():
                y_axis.append((i/num_row_patch)*l - l)
                y_axis.append((i/num_row_patch)*l + l)
                x_axis.append((i%num_row_patch)*l - l)
                x_axis.append((i%num_row_patch)*l + l)

            x_point = round(((max(x_axis)+min(x_axis)+0.0)/2)/490,2)
            y_point = round(((max(y_axis)+min(y_axis)+0.0)/2)/326,2)

            if abs(labels.loc[file][0]/490-x_point)<=0.05 and abs(labels.loc[file][1]/326-y_point)<=0.05:
                accuracy.append(1)
            else:
                accuracy.append(0)           
        else:
            #print (file, 0.50, 0.50)
            accuracy.append(0)
    return accuracy

if __name__ == '__main__':
    path =  sys.argv[1]
    print 'Training the model using Kmeans'
    find_phone_model = train_phone(path)
    print 'Train Model complete'
    print 'Estimating Training accuracy'
    accuracy = train_accuracy(path,find_phone_model)
    print 'Train Accuracy = ', (sum(accuracy)+0.0)/len(accuracy)

    filename = 'find_phone_model.pk'
    with open('./'+filename, 'wb') as file:
        pickle.dump(find_phone_model, file)




