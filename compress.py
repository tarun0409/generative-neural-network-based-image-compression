import sys
import numpy as npy
import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data

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


def Compress_Airplane():
    cifar_gan = AM.CIFAR()
    cifar_gan.generator.load_weights('./airplane_generator_weights.h5')
    cifar_gan.discriminator.load_weights('./airplane_discriminator_weights.h5')
    noise_input = npy.random.uniform(-1.0, 1.0, size=[1, 3072])
    gen_image = cifar_gan.generator.predict(noise_input)
    airfile = open('./airplane_test.pickle', 'rb') 
    x_train = pickle.load(airfile)
    airfile.close()
    r_index = npy.random.randint(0,len(x_train))
    orig_image = x_train[r_index]
    
    Compress_M = Sequential()
    Compress_M.add(Conv2D(1, 8, strides=1, input_shape=(32,32,3),padding='same'))
#     Compress_M.add(LeakyReLU(alpha=0.05))
#     Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Conv2D(1, 8, strides=2,padding='same'))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Activation('sigmoid'))
    #     Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Conv2D(1, 8, strides=2,padding='same'))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.5))

    Compress_M.add(Flatten())
    Compress_M.add(Dense(2048))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.5))
    Compress_M.add(Dense(3072))
#     Compress_M.add(Dense(500,input_shape=784))
    # Compress_M.add(LeakyReLU(alpha=0.05))
#     Compress_M.add(UpSampling2D())
#     Compress_M.add(Conv2DTranspose(1, 15, padding='same'))
#     Compress_M.add(Linear())
    Compress_M.compile(loss='cosine_proximity', optimizer='Adadelta', metrics=['accuracy'])

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
    plt.imsave('./orig_image1.png',orig_image)
    plt.imshow(comp_img)
    plt.imsave('./airplane_comp_img1.png',comp_img)
    plt.show()

def Compress_Automobile():
    cifar_gan = AT.CIFAR()
    cifar_gan.generator.load_weights('./autos_generator_weights.h5')
    cifar_gan.discriminator.load_weights('./autos_discriminator_weights.h5')
    noise_input = npy.random.uniform(-1.0, 1.0, size=[1, 3072])
    gen_image = cifar_gan.generator.predict(noise_input)
    airfile = open('./automobile_test.pickle', 'rb') 
    x_train = pickle.load(airfile)
    airfile.close()
    r_index = npy.random.randint(0,len(x_train))
    orig_image = x_train[r_index]
    Compress_M = Sequential()
    Compress_M.add(Conv2D(1, 8, strides=1, input_shape=(32,32,3),padding='same'))
#     Compress_M.add(LeakyReLU(alpha=0.05))
#     Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Conv2D(1, 8, strides=2,padding='same'))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Activation('sigmoid'))
    #     Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Conv2D(1, 8, strides=2,padding='same'))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.5))

    Compress_M.add(Flatten())
    Compress_M.add(Dense(2048))
#     Compress_M.add(Activation('relu'))
    Compress_M.add(LeakyReLU(alpha=0.5))
    Compress_M.add(Dense(3072))
#     Compress_M.add(Dense(500,input_shape=784))
    # Compress_M.add(LeakyReLU(alpha=0.05))
#     Compress_M.add(UpSampling2D())
#     Compress_M.add(Conv2DTranspose(1, 15, padding='same'))
#     Compress_M.add(Linear())
    Compress_M.compile(loss='cosine_proximity', optimizer='Nadam', metrics=['accuracy'])

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
    plt.imsave('./orig_image1.png',orig_image)
    plt.imshow(comp_img)
    plt.imsave('./auto_comp_img1.png',comp_img)
    plt.show()

def Compress_Flowers():
    flow_gan = FM.Flower_model()
    flow_gan.generator.load_weights('./flowers_generator_weights.h5')
    flow_gan.discriminator.load_weights('./flowers_discriminator_weights.h5')
    noise_input = npy.random.uniform(-1.0, 1.0, size=[1, 3072])
    gen_image = flow_gan.generator.predict(noise_input)
    airfile = open('./flowers_resized_test.pickle', 'rb') 
    x_train = pickle.load(airfile)
    airfile.close()
    r_index = npy.random.randint(0,len(x_train))
    orig_image = x_train[r_index]
    
    Compress_M = Sequential()
    Compress_M.add(Conv2D(1, 8, strides=1, input_shape=(32,32,3),padding='same'))
#     Compress_M.add(LeakyReLU(alpha=0.05))
#     Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Conv2D(1, 8, strides=2,padding='same'))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Activation('sigmoid'))
    #     Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Conv2D(1, 8, strides=2,padding='same'))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.5))

    Compress_M.add(Flatten())
    Compress_M.add(Dense(2048))
    Compress_M.add(Activation('sigmoid'))
#     Compress_M.add(LeakyReLU(alpha=0.5))
    Compress_M.add(Dense(3072))
#     Compress_M.add(Dense(500,input_shape=784))
    # Compress_M.add(LeakyReLU(alpha=0.05))
#     Compress_M.add(UpSampling2D())
#     Compress_M.add(Conv2DTranspose(1, 15, padding='same'))
#     Compress_M.add(Linear())
    Compress_M.compile(loss='cosine_proximity', optimizer='Adadelta', metrics=['accuracy'])

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
    plt.imsave('./orig_image1.png',orig_image)
    plt.imshow(comp_img)
    plt.imsave('./flow_comp_img1.png',comp_img)
    plt.show()
    

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Please provide image path and dataset name')
        sys.exit()
    image_path = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == 'automobiles':
        Compress_Automobile()
    elif dataset == 'airplanes':
        Compress_Airplane()
    elif dataset == 'flowers':
        Compress_Flowers()
        

