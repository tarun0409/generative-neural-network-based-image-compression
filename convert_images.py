from PIL import Image
import numpy
from matplotlib import pyplot as plt
from tempfile import TemporaryFile

import sys
import glob
import errno

width = 64
height = 64 


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (64, 64), (255))  # creates white canvas of 64X64 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva

def resize(img,width,height):
    new_img=img.resize((width, height), Image.ANTIALIAS)
    return new_img

# path = '/home/megha/Documents/second sem/smai/GAN project/ImageToMNIST/path/to/dataset/train/'+sys.argv[1]+'/*.png'  

# print(files)
def convert_to_numpy_array(img):
    arr=numpy.array(img)
    return arr



def open_file(path): 
    file_array=[[]]
    files = glob.glob(path)  
    for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        
        try:
            # with open(name) as f: # No need to specify 'r': this is the default.
                im1 = Image.open(name)
                im2 = resize(im1,width,height)
                #x=name
                arr = convert_to_numpy_array(im2)
                arr_1D=arr.flatten() 
                numpy.append(file_array,arr_1D)
                # x=arr
                # print(name)
                #print(arr)
                # sys.stdout.write(f.read())
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise 
    return file_array


path = '/media/megha/New Volume/smai/102flowers/jpg/*.jpg'
arrays=open_file(path)
outfile = TemporaryFile()# Propagate other kinds of IOError.
numpy.save(outfile, arrays)
# print(x)
# # img = Image.fromarray(x,interpolation='antialias')
# # plt.imshow(img)
# plt.imshow(im2)#, interpolation='nearest')
# plt.show()

# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
# imageFile = "image_00001.jpg"
# im1 = Image.open(imageFile)
# # adjust width and height to your needs
# width = 32
# height = 32
# # use one of these filter options to resize the image
# im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
# # im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
# # im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
# # im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
# ext = ".jpg"
# im2.save("NEAREST" + ext)
# # im3.save("BILINEAR" + ext)
# # im4.save("BICUBIC" + ext)
# # im5.save("ANTIALIAS" + ext)
# # Convert PIL Image to NumPy array
# # img = Image.open("image_00001.jpg")
# arr = numpy.array(im2)
# for i in range(500):
#     for j in range(591):
#         print(arr[i][j])
#         # print(' , ')
#     print('\n')
#print(arr) 
# Convert array to Image
# plt.imshow(arr, interpolation='nearest')
# plt.show()
# img = Image.fromarray(arr)