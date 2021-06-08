import matplotlib.pyplot as plt
from skimage import io
from imgnp import *


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