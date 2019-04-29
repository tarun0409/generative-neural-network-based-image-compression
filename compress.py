import sys
import numpy as npy
import tensorflow as tf
import pickle
import cv2

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import Airplane_model as AM
import Flower_model as FM
import Automobile_model as AT
import compare_images as CMP


def Compress_Airplane():
    cifar_gan = AM.CIFAR()
    cifar_gan.generator.load_weights('./Airplane/airplane_generator_weights.h5')
    cifar_gan.discriminator.load_weights('./Airplane/airplane_discriminator_weights.h5')
    noise_input = npy.random.uniform(-1.0, 1.0, size=[1, 3072])
    gen_image = cifar_gan.generator.predict(noise_input)
    airfile = open('./Airplane/airplane_test.pickle', 'rb') 
    x_train = pickle.load(airfile)
    airfile.close()
    r_index = npy.random.randint(0,len(x_train))
    orig_image = x_train[r_index]
    
    Compress_M = Sequential()
    dimen = 8
    depth = 96
    dropout = 0.4
    Compress_M.add(Dense(dimen*dimen*100,input_dim=3072))
    Compress_M.add(Dense(dimen*dimen*depth))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Reshape((dimen, dimen, depth)))
    Compress_M.add(Dropout(dropout))
    Compress_M.add(UpSampling2D())
    Compress_M.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(UpSampling2D())
    Compress_M.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/16), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/32), 5, padding='same'))
    Compress_M.add(Activation('sigmoid'))
    Compress_M.compile(loss='cosine_proximity', optimizer='Adadelta', metrics=['accuracy'])
    Compress_M.summary()

    x = npy.array(gen_image)
    y = npy.array(orig_image)
    
    print(npy.shape(x))
    print(npy.shape(y))
    
#     x = npy.reshape(x,(28,28,1))
    y = npy.reshape(y,(1,3072))
    
    
    Compress_M.fit(x,y,epochs=150,verbose=1)
    out_img = Compress_M.predict(x)
    print(npy.shape(out_img))
    
    comp_img = npy.reshape(out_img,(32,32,3))
    gen_img = npy.reshape(gen_image,(32,32,3))
    plt.imshow(gen_img)
    plt.show()
    plt.imshow(orig_image)
    plt.show()
    plt.imsave('./Airplane/airplane_orig_image1.png',orig_image)
    plt.imshow(comp_img)
    plt.imsave('./Airplane/airplane_comp_img1.png',comp_img)
    plt.show()

    orig_image = cv2.imread('./Airplane/airplane_orig_image1.png')
    comp_img = cv2.imread('./Airplane/airplane_comp_img1.png')
    mse = CMP.mse(orig_image,comp_img)
    l1_error = CMP.L1(orig_image,comp_img)
    psnr = CMP.psnr(orig_image,comp_img)
    ms_ssim = CMP.ms_ssim(orig_image,comp_img)
    print("Mean Squared Error:",mse)
    print("L1 Error:",l1_error)
    print("PSNR:",psnr)
    print("MS_SSIM:",ms_ssim)

def Compress_Automobile():
    cifar_gan = AT.CIFAR()
    cifar_gan.generator.load_weights('./Automobile/autos_generator_weights.h5')
    cifar_gan.discriminator.load_weights('./Automobile/autos_discriminator_weights.h5')
    noise_input = npy.random.uniform(-1.0, 1.0, size=[1, 3072])
    gen_image = cifar_gan.generator.predict(noise_input)
    airfile = open('./Automobile/automobile_test.pickle', 'rb') 
    x_train = pickle.load(airfile)
    airfile.close()
    r_index = npy.random.randint(0,len(x_train))
    orig_image = x_train[r_index]

    Compress_M = Sequential()
    dimen = 8
    depth = 96
    dropout = 0.4
    Compress_M.add(Dense(dimen*dimen*100,input_dim=3072))
    Compress_M.add(Dense(dimen*dimen*depth))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Reshape((dimen, dimen, depth)))
    Compress_M.add(Dropout(dropout))
    Compress_M.add(UpSampling2D())
    Compress_M.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(UpSampling2D())
    Compress_M.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/16), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/32), 5, padding='same'))
    Compress_M.add(Activation('sigmoid'))
    Compress_M.compile(loss='cosine_proximity', optimizer='Adadelta', metrics=['accuracy'])
    Compress_M.summary()

    x = npy.array(gen_image)
    y = npy.array(orig_image)
    
    print(npy.shape(x))
    print(npy.shape(y))
    
#     x = npy.reshape(x,(28,28,1))
    y = npy.reshape(y,(1,3072))
    
    
    Compress_M.fit(x,y,epochs=50,verbose=1)
    out_img = Compress_M.predict(x)
    print(npy.shape(out_img))
    
    comp_img = npy.reshape(out_img,(32,32,3))
    gen_img = npy.reshape(gen_image,(32,32,3))
    plt.imshow(gen_img)
    plt.show()
    plt.imshow(orig_image)
    plt.show()
    plt.imsave('./Automobile/auto_orig_image1.png',orig_image)
    plt.imshow(comp_img)
    plt.imsave('./Automobile/auto_comp_img1.png',comp_img)
    plt.show()

    orig_image = cv2.imread('./Automobile/auto_orig_image1.png')
    comp_img = cv2.imread('./Automobile/auto_comp_img1.png')
    mse = CMP.mse(orig_image,comp_img)
    l1_error = CMP.L1(orig_image,comp_img)
    psnr = CMP.psnr(orig_image,comp_img)
    ms_ssim = CMP.ms_ssim(orig_image,comp_img)
    print("Mean Squared Error:",mse)
    print("L1 Error:",l1_error)
    print("PSNR:",psnr)
    print("MS_SSIM:",ms_ssim)

def Compress_Flowers():
    flow_gan = FM.Flower_model()
    flow_gan.generator.load_weights('./Flower/flowers_generator_weights.h5')
    flow_gan.discriminator.load_weights('./Flower/flowers_discriminator_weights.h5')
    noise_input = npy.random.uniform(-1.0, 1.0, size=[1, 3072])
    gen_image = flow_gan.generator.predict(noise_input)
    airfile = open('./Flower/flowers_resized_test.pickle', 'rb') 
    x_train = pickle.load(airfile)
    airfile.close()
    r_index = npy.random.randint(0,len(x_train))
    orig_image = x_train[r_index]
    
    Compress_M = Sequential()
    dimen = 8
    depth = 96
    dropout = 0.4
    Compress_M.add(Dense(dimen*dimen*100,input_dim=3072))
    Compress_M.add(Dense(dimen*dimen*depth))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Reshape((dimen, dimen, depth)))
    Compress_M.add(Dropout(dropout))
    Compress_M.add(UpSampling2D())
    Compress_M.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(UpSampling2D())
    Compress_M.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/16), 5, padding='same'))
    Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Conv2DTranspose(int(depth/32), 5, padding='same'))
    Compress_M.add(Activation('sigmoid'))
    Compress_M.compile(loss='cosine_proximity', optimizer='Adadelta', metrics=['accuracy'])
    Compress_M.summary()

    x = npy.array(gen_image)
    y = npy.array(orig_image)
    
    print(npy.shape(x))
    print(npy.shape(y))
    
#     x = npy.reshape(x,(28,28,1))
    y = npy.reshape(y,(1,3072))
    
    
    Compress_M.fit(x,y,epochs=150,verbose=1)
    out_img = Compress_M.predict(x)
    print(npy.shape(out_img))
    
    comp_img = npy.reshape(out_img,(32,32,3))
    gen_img = npy.reshape(gen_image,(32,32,3))
    plt.imshow(gen_img)
    plt.show()
    plt.imshow(orig_image)
    plt.show()
    plt.imsave('./Flower/flow_orig_image1.png',orig_image)
    plt.imshow(comp_img)
    plt.imsave('./Flower/flow_comp_img1.png',comp_img)
    plt.show()

    orig_image = cv2.imread('./Flower/flow_orig_image1.png')
    comp_img = cv2.imread('./Flower/flow_comp_img1.png')
    mse = CMP.mse(orig_image,comp_img)
    l1_error = CMP.L1(orig_image,comp_img)
    psnr = CMP.psnr(orig_image,comp_img)
    ms_ssim = CMP.ms_ssim(orig_image,comp_img)
    print("Mean Squared Error:",mse)
    print("L1 Error:",l1_error)
    print("PSNR:",psnr)
    print("MS_SSIM:",ms_ssim)
    

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print('Please provide a dataset name (automobiles/airplanes/flowers)')
        sys.exit()
    # image_path = sys.argv[1]
    dataset = sys.argv[1]
    if dataset == 'automobiles':
        Compress_Automobile()
    elif dataset == 'airplanes':
        Compress_Airplane()
    elif dataset == 'flowers':
        Compress_Flowers()
        

