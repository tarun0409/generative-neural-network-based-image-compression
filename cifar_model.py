import numpy as npy
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

class DCGAN(object):

    def __init__(self,img_rows=28, img_cols=28, img_chanl=1):
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
        dimen = 7
        depth = 64+64+64+64
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
        # self.Generator.add(Activation('relu'))

        self.Generator.add(Conv2DTranspose(1, 5, padding='same'))
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

        self.Discriminator.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
        self.Discriminator.add(LeakyReLU(alpha=0.05))
        self.Discriminator.add(MaxPooling2D(pool_size=(2,2)))
        self.Discriminator.add(Dropout(0.25))
        self.Discriminator.add(Flatten())
        self.Discriminator.add(Dense(512))
        self.Discriminator.add(LeakyReLU(alpha=0.05))
        self.Discriminator.add(Dropout(0.5))
        # self.Discriminator.add(Dense(NB_CLASSES))
        # self.Discriminator.add(Activation('softmax'))



        # self.Discriminator.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
        # self.Discriminator.add(LeakyReLU(alpha=0.2))
        # self.Discriminator.add(Dropout(dropout))

        # self.Discriminator.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        # self.Discriminator.add(LeakyReLU(alpha=0.2))
        # self.Discriminator.add(Dropout(dropout))

        # self.Discriminator.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        # self.Discriminator.add(LeakyReLU(alpha=0.2))
        # self.Discriminator.add(Dropout(dropout))

        # self.Discriminator.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        # self.Discriminator.add(LeakyReLU(alpha=0.2))
        # self.Discriminator.add(Dropout(dropout))

        # Output: 1-dim probability
        self.Discriminator.add(Flatten())
        # self.Discriminator.add(Dense(28))
        self.Discriminator.add(Dense(1))
        self.Discriminator.add(Activation('sigmoid'))
        print("Discriminator Summary")
        self.Discriminator.summary()
        return self.Discriminator

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

        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(npy.float32)

        self.DCGAN = DCGAN()
        self.generator = self.DCGAN.generator()
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversary = self.DCGAN.adversarial_model()

    def train(self, train_steps=5000, batch_size=256):

        # noise_input = npy.random.uniform(-1,1,size=[16, 784])
        for i in range(train_steps):
            images_train = self.x_train[npy.random.randint(0,self.x_train.shape[0], size=batch_size), :, :, :]
            # print("Images train",npy.shape(images_train))
            noise = npy.random.uniform(-1.0, 1.0, size=[batch_size, 3072]) #changed
            images_gen = self.generator.predict(noise)
            # print("Images Gen",npy.shape(images_gen))
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

            if (i%500==0):
                noisy = npy.random.uniform(-1.0, 1.0, size=[1, 3072]) #changed
                images = self.generator.predict(noisy)
                image = npy.reshape(images, [self.img_rows, self.img_cols])
                plt.imshow(image)
                plt.axis('off')
                plt.imsave('/content/drive/My Drive/Colab Notebooks/MNIST_GAN_misc/gen_mnist.png',image,cmap='gray')
                plt.tight_layout()
                plt.show()
            
        
        
        #Saving weights of model
        self.generator.save_weights('/content/drive/My Drive/Colab Notebooks/MNIST_GAN_misc/generator_weights.h5')
        self.discriminator.save_weights('/content/drive/My Drive/Colab Notebooks/MNIST_GAN_misc/discriminator_weights.h5')


if __name__ == "__main__":
    cifar_dcgan = CIFAR()
    cifar_dcgan.train(train_steps=1001, batch_size=256)


