import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import img_utils as imgutils
from bisect import bisect

#ITERATE THROUGH IMAGE TO DO PIXEL OPERATIONS
def process_img_by_range(img, ops, bins, scale=None):
    
    new_img = np.zeros(shape=img.shape, dtype = int)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            for k in range(len(ops)-1):
                pos = bisect(bins, img[i, j, :][0])
                new_img[i, j, :] = imgutils.get_pixel_operation(ops[pos-1], img[i, j, :], i, j, img.shape, scale)
            
    return new_img

#ITERATE THROUGH IMAGE TO DO ARITHMETIC OPERATIONS BETWEEN TWO IMAGES
def process_img_operations(img1, img2, op):
    
    if img1.shape != img2.shape:
        print('images must have same shape')
        return None
    
    new_img = np.zeros(shape=img.shape, dtype = int)
   
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            
            new_img[i, j, :] = imgutils.get_arithmetic_operation(op, img1[i, j, :], img2[i, j, :])                

    return new_img

#ITERATE THROUGH IMAGE TO DO PIXEL OPERATIONS
def process_img_pixels(img, op, scale=None):
    
    new_img = np.zeros(shape=img.shape, dtype = int)
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            new_img[i, j, :] = imgutils.get_pixel_operation(op, img[i, j, :], i, j, img.shape, scale)
            
    return new_img

#ITERATE THROUGH IMAGE TO TRANSFORM IMAGES BASED ON TRANSFORMATION MATRICES
def process_img_transformations(img, transf):
    
    new_img = np.zeros(shape=img.shape, dtype = int)
    
    affine_matrix = imgutils.get_transf_matrix('identity')
    
    for t in transf[::-1]:
        affine_matrix = np.matmul(affine_matrix, imgutils.get_transf_matrix(t[0], t[1], t[2]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            newpos = np.matmul(affine_matrix, [i, j, 1]).astype(int)
            
            if newpos[0] >= 0 and newpos[0] < new_img.shape[0] and newpos[1] >= 0 and newpos[1] < new_img.shape[1]:
                new_img[newpos[0], newpos[1], :] = img[i, j, :]
    
    return new_img

#READ IMAGE
filename = 'taças.jpg'
img = io.imread(filename)

#CONVERT TO GRAY SCALE
gray_img = process_img_pixels(img, 'gray')
plt.imshow(gray_img)

##EX1
#FADE BASED ON X POS
x_img = process_img_pixels(gray_img, 'xfade')
plt.imshow(x_img)

#SUM GRAYSCALE WITH X FADED
sum_img = process_img_operations(x_img, gray_img, 'sum')
sum_img = imgutils.normalize_img(sum_img)
plt.imshow(sum_img)

#FADE BASED ON Y POS
y_img = process_img_pixels(gray_img, 'yfade')
plt.imshow(y_img)

#SUM GRAYSCALE WITH Y FADED
sum_img = process_img_operations(y_img, gray_img, 'sum')
sum_img = imgutils.normalize_img(sum_img)
plt.imshow(sum_img)

#SUM GRAYSCALE WITH Y FADED AND X FADED
sum_img = process_img_operations(x_img, y_img, 'sum')
sum_img = imgutils.normalize_img(sum_img)
plt.imshow(sum_img)

#AFFINE MATRIX TO ROTATE IMAGE IN THE CENTER
transf = [['translate', img.shape[0]/2, img.shape[1]/2],
          ['rotate', 0.2, 0],
          ['translate', -img.shape[0]/2, -img.shape[1]/2]]
new_img = process_img_transformations(img ,transf)
plt.imshow(new_img)


##EX2
#APPLY NON-LINEAR FUNC EXP
exp_img = process_img_pixels(gray_img, 'exp', 1)
exp_img = imgutils.normalize_img(exp_img)
plt.imshow(exp_img)

#APPLY NON-LINEAR FUNC SQUARE
sqr_img = process_img_pixels(gray_img, 'square', 1)
sqr_img = imgutils.normalize_img(sqr_img)
plt.imshow(sqr_img)

#APPLY NON-LINEAR FUNC SQUARE ROOT
rot_img = process_img_pixels(gray_img, 'root', 1)
rot_img = imgutils.normalize_img(rot_img)
plt.imshow(rot_img)

#APPLY NON-LINEAR FUNC LOG
log_img = process_img_pixels(gray_img, 'log', 1)
log_img = imgutils.normalize_img(log_img)
plt.imshow(log_img)

#BINARIZE IMAGES WITH 128 THRESHOLD 
bin_img = process_img_by_range(gray_img, ['min','max'], [0,128,256])
plt.imshow(bin_img)

#SQUARE AND LOG IMAGES WITH 128 THRESHOLD
nol_img = process_img_by_range(gray_img, ['square','log'], [0,128,256], 1)
plt.imshow(nol_img)

