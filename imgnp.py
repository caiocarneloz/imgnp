import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def normalize_img(img):
    
    max_value = np.max(img)
    new_img = ((img/max_value)*255).astype(int)
    
    return new_img

def gray_scale(pixel, i, j, shape):
    return np.mean(pixel).astype(int)

def x_fade(pixel, i, j, shape):
    return (pixel * (j/shape[1])).astype(int)

def y_fade(pixel, i, j, shape):
    return (pixel * (i/shape[0])).astype(int)

def process_img_pixels(img, func):
    
    new_img = np.zeros(shape=img.shape, dtype = int)
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            new_img[i, j, :] = func(img[i, j, :], i, j, img.shape)
            
    return new_img

def img_sum(pixel1, pixel2):
    return pixel1 + pixel2

def img_sub(pixel1, pixel2):
    return pixel1 - pixel2

def img_div(pixel1, pixel2):
    return pixel1 / pixel2

def img_mul(pixel1, pixel2):
    return pixel1 * pixel2

def process_img_operations(img1, img2, func):
    
    if img1.shape != img2.shape:
        print('images must have same shape')
        return None
    
    new_img = np.zeros(shape=img.shape, dtype = int)
   
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            
            new_img[i, j, :] = func(img1[i, j, :], img2[i, j, :])                

    return new_img


def get_transf_matrix(transf_type, scalar1=0, scalar2=0):
    
    if transf_type == 'rotate':
        return np.array([[np.cos(scalar1), -np.sin(scalar1), 0], [np.sin(scalar1), np.cos(scalar1), 0], [0,0,1]])
    if transf_type == 'translate':
        return np.array([[1, 0, -scalar1], [0, 1, -scalar2], [0 , 0, 1]])
    if transf_type == 'scale':
        return np.array([[1/scalar1, 0, 0], [0, 1/scalar2, 0], [0 , 0, 1]])
    if transf_type == 'shear':
        return np.array([[1, -scalar1, 0], [-scalar2, 1, 0], [0 , 0, 1]])
    if transf_type == 'identity':
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    return False

def process_img_transformations(img, transf):
    
    new_img = np.zeros(shape=img.shape, dtype = int)
    
    affine_matrix = get_transf_matrix('identity')
    
    for t in transf[::-1]:
        affine_matrix = np.matmul(affine_matrix, get_transf_matrix(t[0], t[1], t[2]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            newpos = np.matmul(affine_matrix, [i, j, 1]).astype(int)
            
            if newpos[0] >= 0 and newpos[0] < new_img.shape[0] and newpos[1] >= 0 and newpos[1] < new_img.shape[1]:
                new_img[newpos[0], newpos[1], :] = img[i, j, :]
    
    return new_img

filename = 'taças.jpg'
img = io.imread(filename)

gray_img = process_img_pixels(img, gray_scale)
plt.imshow(gray_img)

x_img = process_img_pixels(gray_img, x_fade)
plt.imshow(x_img)

sum_img = process_img_operations(x_img, gray_img, img_sum)
sum_img = normalize_img(sum_img)
plt.imshow(sum_img)

y_img = process_img_pixels(gray_img, y_fade)
plt.imshow(y_img)

sum_img = process_img_operations(y_img, gray_img, img_sum)
sum_img = normalize_img(sum_img)
plt.imshow(sum_img)


sum_img = process_img_operations(x_img, y_img, img_sum)
sum_img = normalize_img(sum_img)
plt.imshow(sum_img)


transf = [['translate', img.shape[0]/2, img.shape[1]/2],
          ['rotate', 0.2, 0],
          ['translate', -img.shape[0]/2, -img.shape[1]/2]]
new_img = process_img_transformations(img ,transf)
plt.imshow(new_img)