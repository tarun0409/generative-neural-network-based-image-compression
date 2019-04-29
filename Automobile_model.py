import tensorflow as tf
import numpy as npy
import pickle
import datetime

from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

class DCGAN(object):

    def __init__(self,img_rows=32, img_cols=32, img_chanl=3):
        self.image_rows = img_rows
        self.image_columns = img_cols
        self.image_channels = img_chanl
        self.Discriminator = None
        self.Generator = None
        self.Discriminator_Model = None
        self.Adversarial_Model = None
        self.Compressor_Model  = None

    def generator(self):
        if self.Generator:
            return self.Generator
        self.Generator = Sequential()
        dropout = 0.4
        dimen = 8
        depth = 96
        self.Generator.add(Dense(dimen*dimen*depth,input_dim=3072))
        self.Generator.add(LeakyReLU(alpha=0.05))
        # self.Generator.add(Activation('relu'))
        self.Generator.add(Reshape((dimen, dimen, depth)))
        self.Generator.add(Dropout(dropout))

        self.Generator.add(UpSampling2D())
        self.Generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.Generator.add(LeakyReLU(alpha=0.05))
        # self.Generator.add(Activation('relu'))

        self.Generator.add(UpSampling2D())
        self.Generator.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.Generator.add(LeakyReLU(alpha=0.05))
        # self.Generator.add(Activation('relu'))

        self.Generator.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.Generator.add(LeakyReLU(alpha=0.05))
        
        self.Generator.add(Conv2DTranspose(int(depth/16), 5, padding='same'))
        self.Generator.add(LeakyReLU(alpha=0.05))
        # self.Generator.add(Activation('relu'))

        self.Generator.add(Conv2DTranspose(int(depth/32), 5, padding='same'))
#         self.Generator.add(LeakyReLU(alpha=0.05))
        
        self.Generator.add(Activation('sigmoid'))
        print("Generator Summary")
        self.Generator.summary()
        return self.Generator


    def discriminator(self):
        if self.Discriminator:
            return self.Discriminator
        self.Discriminator = Sequential()
        depth = 64
        dropout = 0.4
        input_shape = (self.image_rows, self.image_columns, self.image_channels)

        self.Discriminator.add(Conv2D(6,5,strides=(2,2),padding='same',input_shape=input_shape))
        self.Discriminator.add(LeakyReLU(alpha=0.05))
        self.Discriminator.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.Discriminator.add(Dropout(0.25))
        self.Discriminator.add(Conv2D(6,5,strides=(2,2),padding='same'))
        self.Discriminator.add(LeakyReLU(alpha=0.05))
        self.Discriminator.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.Discriminator.add(Flatten())
        self.Discriminator.add(Dense(120))
        self.Discriminator.add(LeakyReLU(alpha=0.05))
        self.Discriminator.add(Dense(84))
        self.Discriminator.add(LeakyReLU(alpha=0.05))
        self.Discriminator.add(Dense(1))
        self.Discriminator.add(Activation('sigmoid'))
        print("Discriminator Summary")
        self.Discriminator.summary()
        return self.Discriminator
#         

    def discriminator_model(self):
        if self.Discriminator_Model:
            return self.Discriminator_Model
        optimizer = RMSprop(lr=0.0002, decay=6e-8)  
        self.Discriminator_Model = Sequential()
        self.Discriminator_Model.add(self.discriminator())
        self.Discriminator_Model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.Discriminator_Model

    def adversarial_model(self):
        if self.Adversarial_Model:
            return self.Adversarial_Model
        optimizer = RMSprop(lr=0.0001, decay=3e-8)  
        self.Adversarial_Model = Sequential()
        self.Adversarial_Model.add(self.generator())
        self.Adversarial_Model.add(self.discriminator())
        self.Adversarial_Model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.Adversarial_Model


class CIFAR(object):

    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.img_chanl = 3

#         self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
#         self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(npy.float32)
        airfile = open('./Automobile/automobile_train.pickle', 'rb') 
        self.x_train = pickle.load(airfile)
        airfile.close()
        self.x_train = self.x_train.astype(npy.float32)

        self.DCGAN = DCGAN()
        self.generator = self.DCGAN.generator()
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversary = self.DCGAN.adversarial_model()

    def train(self, train_steps=5000, batch_size=256):

        # noise_input = npy.random.uniform(-1,1,size=[16, 784])
        for i in range(train_steps):
            images_train = self.x_train[npy.random.randint(0,self.x_train.shape[0], size=batch_size), :, :, :]
#             print("Images train",npy.shape(images_train))
            if(i==0):
                print(npy.shape(images_train))
                img = images_train[0]/255
#               image = npy.reshape(images_train[0], [self.img_rows, self.img_cols,self.img_chanl])
#               print(images_train[0])
                plt.imshow(img)
                plt.show()
            noise = npy.random.uniform(-1.0, 1.0, size=[batch_size, 3072]) #changed
            images_gen = self.generator.predict(noise)
#             print("Images Gen",npy.shape(images_gen))
            x = npy.concatenate((images_train, images_gen))
            y = npy.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x,y)

            y = npy.ones([batch_size, 1])
            noise = npy.random.uniform(-1.0, 1.0, size=[batch_size, 3072]) #changed
            a_loss = self.adversary.train_on_batch(noise, y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            if (i%100==0):
                noisy = npy.random.uniform(-1.0, 1.0, size=[1, 3072]) #changed
                images = self.generator.predict(noisy)
                image = npy.reshape(images, [self.img_rows, self.img_cols,self.img_chanl])
                plt.imshow(image)
                plt.axis('off')
                plt.imsave('/content/drive/My Drive/smai_cifar/autos_gen_cifar.png',image)
                plt.tight_layout()
                plt.show()
                self.generator.save_weights('/content/drive/My Drive/smai_cifar/autos_generator_weights.h5')
                self.discriminator.save_weights('/content/drive/My Drive/smai_cifar/autos_discriminator_weights.h5')
                print('Weights last saved at :')
                print(datetime.datetime.now())
                
            
        
        
        #Saving weights of model
        self.generator.save_weights('/content/drive/My Drive/smai_cifar/autos_generator_weights.h5')
        self.discriminator.save_weights('/content/drive/My Drive/smai_cifar/autos_discriminator_weights.h5')


if __name__ == "__main__":
    print('Program started at:')
    print(datetime.datetime.now())
    cifar_dcgan = CIFAR()
    cifar_dcgan.train(train_steps=30001, batch_size=256)
    print('Program ended at :')
    print(datetime.datetime.now())