import numpy as npy
import math
import tensorflow as tf

def L1_loss(X, Y):
    n = len(X)
    loss = npy.sum(npy.abs(X-Y))
    loss = loss/n
    return loss

def L1_loss_dash(X,Y):
    return npy.sign(X-Y)

def L2_loss(X,Y):
    n = len(X)
    loss = 0
    for x, y in zip(X,Y):
        loss += math.pow((x-y),2)
    loss = loss/(2*n)
    return loss

def L2_loss_dash(X,Y):
    n = len(X)
    loss = 0
    for x,y in zip(X,Y):
        loss += x-y
    loss = loss/n
    return loss

def SSIM_Loss(X,Y):
    loss = 1 - tf.image.ssim(X,Y)
    return loss

def MS_SSIM_Loss(X,Y):
    loss = tf.image.ssim_multiscale(X,Y)
    return loss

def Mix_Loss(X,Y):
    alpha = 0.84
    Xarr = npy.array(X)
    Yarr = npy.array(Y)
    loss = alpha*MS_SSIM_Loss(X,Y) + (1-alpha)*Gauss_coeff*L1_loss((Xarr,Yarr))
    return loss

