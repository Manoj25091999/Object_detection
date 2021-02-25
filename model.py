# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:38:23 2021

@author: Manoj Kumar
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the required files
frozen_graph = 'F:/opencv/frozen_inference_graph.pb'
config_file = 'F:/opencv/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Loading the pre-trained model

model = cv2.dnn_DetectionModel(frozen_graph,config_file)

# Reading the label file

Label = []
with open('label.txt', 'rt') as lbl:
    Label = lbl.read().rstrip('\n').split('\n')
    #label.append(labels)


print(Label)

print(len(Label))


# Object detection in an image 

# Reading an image
cv2.imread('F:/opencv/5acd020c689875bc368b4e57.jpg')




