import sys
import numpy as npy
import tensorflow as tf

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
import MNIST_model as MM


def Compress_MNIST():
    mnist_gan = MM.MNIST()
    mnist_gan.generator.load_weights('./generator_weights.h5')
    mnist_gan.discriminator.load_weights('./discriminator_weights.h5')
    noise_input = npy.random.uniform(-1.0, 1.0, size=[1, 784])
    gen_image = mnist_gan.generator.predict(noise_input)
    orig_image = Image.open('./orig_mnist.png').convert('L')
    
    Compress_M = Sequential()
    Compress_M.add(Conv2D(8, 5, strides=1, input_shape=(28,28,1),padding='same'))
    Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Conv2D(8, 5, strides=2,padding='same'))
#     Compress_M.add(MaxPooling2D(3,1,padding='same'))
    Compress_M.add(Flatten())
    # Compress_M.add(Dense(500,input_shape=784))
    # Compress_M.add(LeakyReLU(alpha=0.05))
    Compress_M.add(Dense(784))
#     Compress_M.add(Linear())
    Compress_M.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

    x = npy.array(gen_image)
    y = npy.array(orig_image)
    
    print(npy.shape(x))
    print(npy.shape(y))
    
#     x = npy.reshape(x,(28,28,1))
    y = npy.reshape(y,(1,784))
    
    
    Compress_M.fit(x,y,epochs=100)
    out_img = Compress_M.predict(x)
    print(npy.shape(out_img))
    
    comp_img = npy.reshape(out_img,(28,28))
    gen_img = npy.reshape(gen_image,(28,28))
    plt.imshow(gen_img,cmap='gray')
    plt.show()
    plt.imshow(orig_image,cmap='gray')
    plt.show()
    plt.imshow(comp_img,cmap='gray')
    plt.imsave('./comp_img.png',comp_img,cmap='gray')
    plt.show()

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Please provide image path and dataset name')
        sys.exit()
    image_path = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == 'mnist':
        Compress_MNIST()
    else:    
        generator_weights_path = dataset + '_generator_weights.h5'
        discriminator_weights = dataset + '_discriminator_weights.h5'
        model_obj = None
        orig_image = Image.open(image_path)
        dims = np.shape(np.asarray(orig_image))
        tot_dim = 1
        for d in dims:
            tot_dim *= d
        

