from PIL import Image
import numpy
from matplotlib import pyplot as plt
from tempfile import TemporaryFile

import sys
import glob
import errno

width = 32
height = 32 


def resize(img,width,height):
    new_img=img.resize((width, height), Image.ANTIALIAS)
    return new_img

def convert_to_numpy_array(img):
    arr=numpy.array(img)
    return arr

def open_file(path): 
    file_array=[[]]
    files = glob.glob(path) 
    i=1 
    for name in files: 
        
        try:
            
                img_name=""
                im1 = Image.open(name)
                im2 = resize(im1,width,height)
                img_name='img'+str(i)+'.png'
                plt.imsave(img_name,im2)
                i+=1
                arr = convert_to_numpy_array(im2)
                arr_1D=arr.flatten() 
                numpy.append(file_array,arr_1D)

        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise 
    return file_array

path = sys.argv[1]+'/*.jpg'
arrays=open_file(path)
numpy.save('outfile', arrays)
