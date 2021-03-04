#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
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


# In[4]:


Label[1]


# In[5]:


# Object detection in an image 

# Reading an image
img = cv2.imread('F:/opencv/5acd020c689875bc368b4e57.jpg')

# Showing the image
plt.imshow(img); #bgr


# In[6]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); # Changing to RGB color format


# In[7]:


# Setting the input according to model architecture

model.setInputSize(320,320)
model.setInputScale(1.0/127.5) # 255/2 = 127.5
model.setInputSwapRB(True) # To automatically convert from bgr to RGB
model.setInputMean((127.5,127.5,127.5))


# In[8]:


# Modelling

classindex, confidence, bbox = model.detect(img, confThreshold=0.6)


# In[9]:


print(classindex)


# In[10]:


print(confidence)


# In[11]:


print(bbox)


# In[12]:


bbox[0]+20


# In[13]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

for classind, conf, boxes in zip(classindex.flatten(),confidence.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2) #bgr
    cv2.putText(img,Label[classind-1].upper(),(boxes[0]+10,boxes[1]+30),font, 
                fontScale=font_scale, color=(0,255,0), thickness=3)
    cv2.putText(img,str(round(conf*100,2)),(boxes[0]+200,boxes[1]+30),font, 
                fontScale=font_scale, color=(0,255,0), thickness=3)


# In[14]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));


# ## Video detection

# In[15]:


video = cv2.VideoCapture('22.mp4')

video.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

# Checking if the video is working properly
if not video.isOpened():
    video=cv2.VideoCapture(0)
    raise IOError('Cannot open video')
    
font_scale=3
font = cv2.FONT_HERSHEY_PLAIN

while True: 
    ret,frame= video.read()
    
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.7)
    print(ClassIndex)
    if(len(Label)!=0): #Checking if the labels are present
        for classind, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(), bbox):
            if(len(Label)<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2) #bgr
                cv2.putText(frame,Label[classind-1].upper(),(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale, color=(0,255,0), thickness=3)
    
    # displaying predictions
    cv2.imshow('Object detection in the video', frame)
    
    ## press q to quit
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# close the video
video.release()
cv2.destroyAllWindows()


# ## Webcam detection

# In[ ]:


cap = cv2.VideoCapture(0)

# Checking if the video is working properly
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
    raise IOError('Cannot open webcam')
    
font_scale=3
font = cv2.FONT_HERSHEY_PLAIN

while True: 
    ret,frame= cap.read()
    
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.7)
    print(ClassIndex)
    if(len(Label)!=0): #Checking if the labels are present
        for classind, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(), bbox):
            if(len(Label)<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2) #bgr
                cv2.putText(frame,Label[classind-1].upper(),(boxes[0]+10,boxes[1]+40),font, 
                            fontScale=font_scale, color=(0,255,0), thickness=3)
                cv2.putText(frame,str(round(conf*100,2)),(boxes[0]+200,boxes[1]+40),font, 
                            fontScale=font_scale, color=(0,255,0), thickness=3)
    
    # displaying predictions
    cv2.imshow('Object detection in the webcam', frame)
    
    ## press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# closing the webcam
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Using YOLO network

# In[16]:


# Loading Yolo Network
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')


# In[17]:


# Loading coco labels
classes=[]
with open('coco.names','r') as c:
    for line in c:
        classes.append(line.strip())


# In[18]:


classes


# In[19]:


layer_names = net.getLayerNames()


# In[21]:


net.getUnconnectedOutLayers()


# In[22]:


output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()] #Loading output layers to show the detected result
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# In[23]:


image = cv2.imread('family-room-1-1588080207.jpg')
plt.imshow(image)


# In[24]:


image = cv2.resize(image, None, fx=0.4, fy=0.4) # Resizing the image
height, width, channels=image.shape


# In[25]:


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #bgr to RGB


# ### Detecting objects from the image

# In[26]:


# We can’t use right away the full image on the network, first we need it to convert it into a blob. 
# Blob is used to extract feature from the image and to resize them.

blob = cv2.dnn.blobFromImage(image,0.00392, (416,416), (0,0,0), True, crop=False)


# In[27]:


# Watching the contents of blob
for b in blob:
    for n,img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)

cv2.waitKey(100)
cv2.destroyAllWindows()


# In[28]:


# Now sending the processed blob image into the yolo algorithm
net.setInput(blob)
outs=net.forward(output_layers)
print(outs)  # Printing out the detections


# In[37]:


# Showing information on the screen

class_ids=[]
confidences=[]
boxes=[]

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.6:
            # Object detected
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            # Rectangle coordinates
            x = int(center_x-w/2)
            y = int(center_y-h/2)
            
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
        


# In[38]:


# When we perform the detection, it happens that we have more boxes for the same object. 
# so we should use another function to remove this “noise”.
# It’s called Non maximum suppresion.

indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.7,0.4)


# In[39]:


indexes


# In[41]:


# Finally showing all the informations on the screen
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(image,(x,y),(x+w,y+h), color, 2)
        cv2.putText(image,label,(x,y+30),font,3,color,3)

cv2.imshow('Image',image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




