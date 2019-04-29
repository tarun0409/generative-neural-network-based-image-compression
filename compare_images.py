
from skimage import measure #import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import sys, os, numpy, scipy.misc, math
from scipy.ndimage import filters
import tensorflow as tf



#!/usr/bin/python
def mse(imageA, imageB):
	error = numpy.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	error /= float(imageA.shape[0] * imageA.shape[1])
	return error


def ms_ssim(imageA, imageB):
	error = tf.image.ssim_multiscale(imageA, imageB,1000000)#measure.compare_ssim(imageA, imageB)
	return error

def psnr(imageA, imageB): # higher the value, lesser the loss
	mserr = mse(imageA,imageB)
	if mserr == 0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mserr))

def L1(imageA, imageB):
	return numpy.sum(numpy.absolute(imageA.astype("float") - imageB.astype("float")))
	




original = cv2.imread(sys.argv[1])
compressed = cv2.imread(sys.argv[2])
im1 = tf.image.decode_png(sys.argv[1])
im2 = tf.image.decode_png(sys.argv[2])
l1_error = L1(original,compressed)
mse_error = mse(original,compressed)
ms_ssim_error = ms_ssim(im1,im2)
psnr_error=psnr(original,compressed)
print(mse_error,psnr_error,ms_ssim_error,l1_error)
#mssim_error=MSSIM().compute(original, compressed)
