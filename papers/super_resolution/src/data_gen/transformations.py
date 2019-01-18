import numpy as np
from PIL import Image
import tensorflow as tf 
import cv2
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from blend_modes import blend_modes

def blur(sess, image,filterSize):

	#global sess ;
	element = 1/(float(filterSize*filterSize))
	W = np.zeros(shape=(filterSize,filterSize,1,1), dtype=np.float32) + element    
	xImage = tf.reshape(image, [-1, image.shape[0], image.shape[1], 1])    
	outImage = tf.nn.conv2d(xImage, W, strides=[1, 1, 1,1], padding='VALID')
	out = sess.run(outImage)
	out = out.reshape((image.shape[0]-filterSize + 1,image.shape[1] - filterSize + 1))
	out  = out.astype(np.uint8)
	return np.expand_dims(out,0) ;

def blend(fg , bg) :

	background_img_raw = bg.resize((fg.shape[1],fg.shape[0])) ;#Image.open(bg_path).convert("RGBA").resize((fg.shape[1],fg.shape[0]))  # RGBA image
	background_img = np.array(background_img_raw)  # Inputs to blend_modes need to be numpy arrays.
	bg = background_img.astype(float)  # Inputs to blend_modes need to be floats.

	fg = fg.astype(float)
	opacity = 0.7  # The opacity of the foreground that is blended onto the background is 70 %.
	blended = np.uint8(blend_modes.soft_light(bg, fg, opacity)) ;
	return blended ;

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))   
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    dist_img = map_coordinates(image, indices, order=1, mode='reflect')
    dist_img = np.uint8(dist_img.reshape(image.shape)) 
    dist_img = Image.fromarray(dist_img)#.convert("L") 

    return np.expand_dims(dist_img,0) ;


def vertical(img,back):
	#print img.shape,back.shape
	h,w= img.shape
	no = 5 ;
	if w > 30 :
		no = 10 ;
	points = []
	# for n in range(1,w):
	# 	points.append(n*no)
	cut_col = no ;
		# print points,len(points)
	#print w , strokes
	# choose_one = random.sample(range(0,w-2),strokes[0])
	back_resize = cv2.resize(back,(w,h))
	i = False ;
	while cut_col < w - 1 :
		#print points[i],img.shape,i
		img[:,cut_col]= back_resize[:,cut_col]
		if i :
			i = False 
			#print i,"here after",points[i]
			img[:,cut_col - 1]= back_resize[:,cut_col - 1]	
		else :
			i = True

		cut_col += no ;
	return img


def remove_shadow(img) :

	dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8)) 
	bg_img = cv2.medianBlur(dilated_img, 21)
	diff_img = 255 - cv2.absdiff(img, bg_img)
	norm_img = diff_img.copy() 
	cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	_, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
	cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

	return thr_img ;
