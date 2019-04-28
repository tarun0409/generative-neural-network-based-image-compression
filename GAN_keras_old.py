import numpy as np
import time
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from skimage import data


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.CM = None #Compressor Model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM
      
    def compressor_model(self):
        if self.CM:
            return self.CM
        optimizer = SGD(lr=0.99, decay=0.005)
        self.CM = Sequential()
        self.CM.add(self.generator())
        self.CM.compile(loss='mean_squared_error',optimizer=optimizer,\
                       metrics=['accuracy'])
        return self.CM
        
        

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32)
#         print("Shape of orig:",np.shape(self.x_train[0]))
#         orig_img = self.x_train[0]
#         orig_img = np.reshape(orig_img,(28,28))
#         plt.imsave('/content/drive/My Drive/Colab Notebooks/GAN_Compress/orig_mnist.png',orig_img,cmap='gray')
        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()
        self.compressor = self.DCGAN.compressor_model()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        self.generator.load_weights('/content/drive/My Drive/Colab Notebooks/GAN_Compress/generator_weights.h5')
        self.discriminator.load_weights('/content/drive/My Drive/Colab Notebooks/GAN_Compress/discriminator_weights.h5')
        noise_input = np.random.uniform(-1.0, 1.0, size=[1, 100])
        gen_img = self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input)[0]
#         cv2.imwrite('/content/drive/My Drive/Colab Notebooks/GAN_Compress/gen_mnist_5.png',gen_img)
#         scipy.misc.imsave('/content/drive/My Drive/Colab Notebooks/GAN_Compress/gen_mnist_5.png',gen_img)
#         plt.savefig('/content/drive/My Drive/Colab Notebooks/GAN_Compress/mnist_5.png')
        orig_img = mpimg.imread('/content/drive/My Drive/Colab Notebooks/GAN_Compress/orig_mnist.png')
        print("Original shape",np.shape(orig_img))
        x = gen_img
#         x = gen_img[:,:,0]
#         print(x)
        y = orig_img #.flatten()
        print("Orig shape",np.shape(y))
#         print(y)
#         y = np.reshape(y,(y.shape[0],1))
        for i in range(100):
            x=np.reshape(x,(28, 28));
#             x = x.flatten()
#             x = np.reshape(x,(x.shape[0],1))
#             print(x.shape)
#             print(y.shape)
            c_loss = self.compressor.train_on_batch(x,y)
#             img = np.reshape(x,(1,28,28,1))
            x = self.compressor.predict(img)
        
        plt.imshow(x, cmap='gray')
#         noise_input = None
#         if save_interval>0:
#             noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
#         for i in range(train_steps):
#             images_train = self.x_train[np.random.randint(0,
#                 self.x_train.shape[0], size=batch_size), :, :, :]
#             noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
#             images_fake = self.generator.predict(noise)
#             x = np.concatenate((images_train, images_fake))
#             y = np.ones([2*batch_size, 1])
#             y[batch_size:, :] = 0
#             d_loss = self.discriminator.train_on_batch(x, y)

#             y = np.ones([batch_size, 1])
#             noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
#             a_loss = self.adversarial.train_on_batch(noise, y)
#             log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
#             log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
#             print(log_mesg)
#             if save_interval>0:
#                 if (i+1)%save_interval==0:
#                     self.plot_images(save2file=True, samples=noise_input.shape[0],\
#                         noise=noise_input, step=(i+1))
#         self.generator.save_weights('/content/drive/My Drive/GAN_Compress/generator_weights.h5')
#         self.discriminator.save_weights('/content/drive/My Drive/GAN_Compress/discriminator_weights.h5')

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            print("Image#",i)
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.imsave('/content/drive/My Drive/Colab Notebooks/GAN_Compress/gen_mnist.png',image,cmap='gray')
#             cv2.imwrite('/content/drive/My Drive/Colab Notebooks/GAN_Compress/gen_mnist.png',image)
            print(np.shape(image))

        plt.tight_layout()
#         if save2file:
#             plt.imsave('/content/drive/My Drive/Colab Notebooks/GAN_Compress/gen_mnist_5.png',images)
#             plt.close('all')
#         else:
        plt.show()
        return images

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=1000, batch_size=256, save_interval=500)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
# mnist_dcgan.plot_images(fake=False, save2file=False)