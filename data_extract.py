import pickle
import os
import numpy as npy
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import cv2
folders = ['airplane','automobile','flowers_resized','mnist']
data = None
for folder_name in folders:
  if folder_name == 'mnist':
    data = input_data.read_data_sets("mnist", one_hot=True).train.images
    data = data.reshape(-1, 28, 28, 1).astype(npy.float32)
    print('Collected all images from mnist')
  else:
    data = []
    folder = '/content/drive/My Drive/'+folder_name
    for filename in os.listdir(folder):
      img = Image.open(os.path.join(folder,filename))
      img_arr = npy.asarray(img)
      img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)
      data.append(img_arr)
    data = npy.array(data)
    print('Collected all images from '+folder_name)
  n = data.shape[0]
  train_n = int(n*0.8)
  X_train = data[:train_n,:,:]
  X_test = data[train_n:,:,:]
  train_file = '/content/drive/My Drive/smai_cifar/'+folder_name+'_train.pickle'
  test_file = '/content/drive/My Drive/smai_cifar/'+folder_name+'_test.pickle'
  train_dump_file = open(train_file,'wb')
  test_dump_file = open(test_file,'wb')
  pickle.dump(X_train,train_dump_file)
  pickle.dump(X_test,test_dump_file)
  train_dump_file.close()
  test_dump_file.close()
  print(train_file+' saved successfully')
  print(test_file+' saved successfully')
  