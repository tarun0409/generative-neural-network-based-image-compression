import numpy as npy
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from keras.models import Sequential


class GAN(object):
    def __init__(self,image_x=28,image_y=28,img_chan=1):
        self.image_rows = image_x
        self.image_cols = image_y
        self.image_channel = img_chan
        self.Disc = None
        self.Gen = None
        self.AdvMod = None
        self.DisMod = None

    def discriminator(self):
        if(self.Disc):
            return self.Disc
        
        inp_shape = (self.image_rows,self.image_cols,self.image_channel)
        channel_0, channel_1, channel_2, channel_3, fc_channel, num_classes = 60, 48, 36, 24, 20, 1
        initializer = tf.variance_scaling_initializer(scale=2.0)
        layers = [
            tf.layers.Conv2D(input_shape=inp_shape,filters=channel_0,kernel_size=5,strides=1,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
            tf.layers.MaxPooling2D(pool_size=2,strides=2,padding="same"),
        tf.layers.Conv2D(filters=channel_1,kernel_size=5,strides=1,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
            tf.layers.MaxPooling2D(pool_size=2,strides=2,padding="same"),
            tf.layers.Conv2D(filters=channel_2,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
            tf.layers.MaxPooling2D(pool_size=2,strides=2,padding="same"),
            tf.layers.Conv2D(filters=channel_3,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
            tf.layers.MaxPooling2D(pool_size=2,strides=2,padding="same"),
            tf.layers.Flatten(),
            tf.layers.Dense(units=fc_channel,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer,activation=tf.nn.relu),
            tf.layers.Dense(units=num_classes,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer,activation=tf.nn.sigmoid)
        ]
        self.Disc = tf.keras.Sequential(layers)
        return self.Disc

    def generator(self):
        if(self.Gen):
            return self.Gen

        depth = 256
        dim = 7
        inp_shape = (dim,dim,depth)
        initializer = tf.variance_scaling_initializer(scale=2.0)
        layers = [
            # tf.layers.Dense(inp_shape,input_dim=100,activation=tf.nn.relu),
            tf.layers.Dense(units=100,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer,activation=tf.nn.relu),
            tf.keras.layers.UpSampling2D(),
            tf.layers.Conv2DTranspose(int(depth/2),5,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
            tf.keras.layers.UpSampling2D(),
            tf.layers.Conv2DTranspose(int(depth/4),5,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
            tf.keras.layers.UpSampling2D(),
            tf.layers.Conv2DTranspose(int(depth/8),5,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
            tf.layers.Conv2DTranspose(1,5,padding="same",activation=tf.nn.sigmoid,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),     
        ]
        self.Gen = tf.keras.Sequential(layers)
        return self.Gen
        
        
    def discriminator_model(self):
        if(self.DisMod):
            return self.DisMod
        
        learn_rate = 1e-4
        optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
        self.DisMod = tf.keras.Sequential()
        self.DisMod.add(self.discriminator())
        self.DisMod.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return self.DisMod

    def adversarial_model(self):
        if(self.AdvMod):
            return self.AdvMod
        
        learn_rate = 1e-4
        optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
        self.AdvMod = tf.keras.Sequential()
        self.AdvMod.add(self.generator())
        self.AdvMod.add(self.discriminator())
        self.AdvMod.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return self.AdvMod

class MNIST_GAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.input_data = input_data.read_data_sets("MNIST_data/",one_hot=True)
        self.input_images = input_images = self.input_data.train.images
        self.input_images = self.input_images.reshape(-1, self.img_rows,self.img_cols, self.channels).astype(npy.float32)
        # print(npy.shape(input_images))
        self.GAN_model = GAN()
        self.discriminator =  self.GAN_model.discriminator_model()
        self.adversarial = self.GAN_model.adversarial_model()
        self.generator = self.GAN_model.generator()

    def train(self,batch_size=256):
        for i in range(0,1000):
            train_img = self.input_images[npy.random.randint(0,self.input_images.shape[0], size=batch_size), :, :, :]
            noise = npy.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            gen_images = self.generator.predict(noise)
            x = npy.concatenate((train_img, gen_images))
            y = npy.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)
            y = npy.ones([batch_size, 1])
            noise = npy.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

if __name__ == "__main__":
    minst_gan = MNIST_GAN()
    batch_size = 256
    minst_gan.train(batch_size)
    # mnist.gan.plot_orig()
    # minst_gan.plot_gen()


    


    
    





