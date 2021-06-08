import numpy as np

#GET MATRICES THAT TRANSFORMS THE SPACE
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

#DO COLOR/INTENSITY OPERATION WITH PIXELS
def get_pixel_operation(op, pixel, i, j, shape, scale=None):
    
    if op == 'gray':
        return np.mean(pixel).astype(int)
    elif op == 'xfade':
        return (pixel * (j/shape[1])).astype(int)
    elif op == 'yfade':
        return (pixel * (i/shape[0])).astype(int)  
    elif op ==  'square':
        return scale*(pixel*pixel.mean().astype(int))
    elif op == 'root':
        return scale*np.sqrt(pixel.mean().astype(int))
    elif op == 'log':
        return scale*np.log10(pixel.mean().astype(int) +1)
    elif op == 'exp':
        return scale*np.exp((pixel.mean()*0.1).astype(int)) - 1
    elif op == 'min':
        return 0
    elif op == 'max':
        return 255

    
    

#DO ARITHMETIC OPERATIONS WITH PIXELS
def get_arithmetic_operation(op, pixel1, pixel2):
    
    if op == 'sum':
        return pixel1 + pixel2
    if op == 'sub':
        return pixel1 - pixel2
    if op == 'div':
        return pixel1 / pixel2
    if op == 'mul':
        return pixel1 * pixel2

#NORMALIZE A GIVEN IMAGE TO HAVE [0, 255] RANGE
def normalize_img(img):
    
    min_value = np.min(img)
    img += min_value
    max_value = np.max(img)
    
    new_img = ((img/max_value)*255).astype(int)
    
    return new_img