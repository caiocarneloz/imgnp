import numpy as np
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
    
    new_img = np.zeros(shape=img1.shape, dtype = int)
   
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