from re import I
import cv2
from util import get_detection
import numpy as np
import random

## 1. Define paths
cfg_path = r'C:\Users\Fame\Desktop\Projects_2024\Computer_Vision_Intermediate_Projects\Semantic_segmentation_with_Tensorflow_OpenCV_in_Python\models\mask_rcnn_inception\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
class_name_path = r'C:\Users\Fame\Desktop\Projects_2024\Computer_Vision_Intermediate_Projects\Semantic_segmentation_with_Tensorflow_OpenCV_in_Python\models\mask_rcnn_inception\mscoco_labels.names'
weights_paths = r'C:\Users\Fame\Desktop\Projects_2024\Computer_Vision_Intermediate_Projects\Semantic_segmentation_with_Tensorflow_OpenCV_in_Python\models\mask_rcnn_inception\frozen_inference_graph.pb'

img_path = r'C:\Users\Fame\Desktop\Projects_2024\Computer_Vision_Intermediate_Projects\Semantic_segmentation_with_Tensorflow_OpenCV_in_Python\cat.png'

## 2. Load Image
img = cv2.imread(img_path)
H, W, C = img.shape

## 3. Load Model
net = cv2.dnn.readNetFromTensorflow(weights_paths, cfg_path)

## 4. Convert Image into blob
blob = cv2.dnn.blobFromImage(img)

## 5. Get masks
boxes, masks = get_detection(net, blob)

## 6. Draw Masks
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(90)]
empty_img = np.zeros((H, W, C))

print(len(masks))
detection_th = 0.5

for j in range(len(masks)):
    bbox = boxes[0, 0, j]
    
    
    class_id = bbox[1]
    score = bbox[2]
    
    if score > detection_th: 
        mask = masks[j]
        x1, y1, x2, y2 = int(bbox[3]*W), int(bbox[4]*H), int(bbox[5]*W), int(bbox[6]*H)
        
        mask=mask[int(class_id)]
        mask = cv2.resize(mask, (x2-x1, y2-y1))
        
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        
        for c in range(3):
            empty_img[y1:y2, x1:x2, c] = mask * colors[int(class_id)][c]
        
      
        ## cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        ## cv2.imshow('img', img)
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        # print(x1, y1, x2, y2)
        
        mask = mask[int(class_id)]
        
        # (90, 15, 15) 90 different images of size 15x15.
        ## print(bbox.shape)
        ## print(mask.shape)
        ## print(H, W)
         

## 7. Visualization

overlay = ((0.6*empty_img)+(0.4*img)).astype('uint8')
        
cv2.imshow('mask', empty_img)
cv2.imshow('img', img)
cv2.imshow('overlay', overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()


