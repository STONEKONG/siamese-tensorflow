import numpy as np
import os
import random
import cv2
from glob import glob 

def eval_classfier(predicts, labels):
    TP = 0 
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(predicts)):
        pre = predicts[i]
        label = labels[i]
        if label == 1 and pre == 1:
            TP += 1
        elif label == 1 and pre == 0:
            FN += 1
        elif label == 0 and pre == 1:
            FP += 1
        else:
            TN += 1  
    print(TP, FN, FP, TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    acc = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, acc

def data_load(input_dir, image_size):

    image_names = os.listdir(input_dir)
    random.shuffle(image_names)
    images = []
    labels = []
    for image_name in image_names:
        img = cv2.imread(os.path.join(input_dir, image_name))
        img = cv2.resize(img, (image_size, image_size))
        images.append(img)
        if 'NG' in image_name:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(images), np.array(labels)

def data_load_2(input_dir, image_size):
    img_class_list = os.listdir(input_dir)

    img_path_list = []
    for img_class in img_class_list:
        img_path_list += glob(os.path.join(input_dir, img_class, '*.jpg'))
    random.shuffle(img_path_list)
    images = []
    labels = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        images.append(img)
        if '/NG/' in img_path:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(images), np.array(labels)