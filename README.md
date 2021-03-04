# Furniture detection using opencv
Here in this project my main aim was to detect different furniture objects (chair, sofa, table etc.) using different deeplearning frameworks, I have employed all the steps from setting the enviornment to detect the required objects via transfer learning. 

I have used two pre-trained models here: 
1) SSDMobileNet-V3 
2) YOLO (You only look once)

The reason for choosing the above models is because of their high speed in comparison to other frameworks like RNN, Fast RNN etc. 

I compared the performance of the above models on the basis of their capibilities i.e., I loaded a sample image and trained both of my models on it while keeping the confidence level value for each of them equal and then I watched which model was detecting more objects, now what I found is that SSDMobileNet was detecting fewer objects in comparison to YOLO for the same confidence level on my sample image. hence I decided to go with YOLO as my final object detection model.

So in this project I have implemented YOLO to detect furniture objects and then tuned its confidence level to increase its accuracy.
