import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dataset_path = os.listdir('room_dataset')

room_types = os.listdir('room_dataset')
print(room_types)
print('Types of rooms found: ', len(room_types))

rooms = []

for item in room_types:
  ## get all the file names
  all_rooms = os.listdir('room_dataset' + '/' +item)

  ## Add the, to the list
  for room in all_rooms:
    rooms.append((item, str('room_dataset' + '/' +item) + '/' + room))

rooms

## Build Dataframe
rooms_df = pd.DataFrame(data=rooms, columns=['room types', 'path'])
rooms_df.head()

## Let's check many samples for each category are present.
print('Total number of rooms in the dataset: ', len(rooms_df))

room_count = rooms_df['room types'].value_counts()
print(room_count)

import cv2
path = 'room_dataset/'

im_size = 64

images = []
labels = []

for i in room_types:
    data_path = path + str(i)  # entered in 1st folder and then 2nd folder and then 3rd folder
    filenames = [i for i in os.listdir(data_path) ]
   # print(filenames)  # will get the names of all images
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)  # reading that image as array
        #print(img)  # will get the image as an array
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)
print(labels)  