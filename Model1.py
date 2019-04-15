# Model 1 using custom keras sequential model

import tensorflow as tf



def model_init_fn(inputs, is_training):
    model = None
    inp_shape = (32,32,3)
    channel_0, channel_1, channel_2, channel_3, num_classes = 60, 48, 36, 24, 10
    #initializer = tf.keras.initializers.lecun_uniform()
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
        #tf.layers.Conv2D(filters=channel_4,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
        tf.layers.Flatten(),
        tf.layers.Dense(units=20,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer),
        tf.layers.Dense(units=num_classes,use_bias=True,kernel_initializer=initializer,bias_initializer=initializer)
    ]
    
    net = tf.keras.Sequential(layers)
    return net(inputs)

learn_rate = 1.5e-3

def optimizer_init_fn():
    optimizer = None
    #pass
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate,momentum=0.98,use_nesterov=True)
    #optimizer = tf.train.AdagradOptimizer(learning_rate = learn_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
    return optimizer

device = '/cpu:0'
print_every = 700
num_epochs = 10
# Add training method
train_data(model_init_fn, optimizer_init_fn, num_epochs)  