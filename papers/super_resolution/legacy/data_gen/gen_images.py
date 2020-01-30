from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.preprocessing import image
import transformations as tfs
import os
# import tensorflow as tf
import random
import traceback, sys
from multiprocessing import Process
from sklearn.utils import shuffle
import shutil
import cv2

is_gray = False ;
bg_raw_imgs = {} ;

def generate_word_image(word, bg, font_name, font_size, rotation, elas_dist, shape, symbol) :

	word = word.decode('utf8')
	if bg == '0.png' :
		bg_color = (0, 0, 0) ;
		fg_color = (255, 255, 255) ;
	else : # black otherwise
		bg_color = (255, 255, 255) ;
		fg_color = 0 ;

	img = Image.new('RGBA', (2000, 2000), bg_color) ;
	draw = ImageDraw.Draw(img) ;
	font = ImageFont.truetype('fonts/all_fonts/' + font_name, font_size) ;

	# draw word on the image
	draw.text((10, 10), word, fill=fg_color, font=font) ;

	# bounding rectangle surrounding text
	size = draw.textsize(word, font=font) ;
	offset = font.getoffset(word)

	# underline/ border
	if shape != 'none' :

		if shape == 'underline' :
			draw.line([(10, 10 + size[1]), (10 + size[0], 10 + size[1])], fill=fg_color) ;
			img = img.crop((10, 8 + offset[1], 10 + size[0], 11 + size[1] + 3)) # increasing height of bounding box by 3 rows(to make underline visible)
								# 2 column gap for top and 1 column gap for bottom(values 8 & 11)
		elif shape == 'left_line' :
			draw.line((10, 10, 10, 10 + size[1]), fill=fg_color, width=4) ;
			img = img.crop((4, 8 + offset[1], 10 + size[0], 11 + size[1]))

		elif shape == 'right_line' :
			draw.line((10 + size[0], 10, 10 + size[0], 10 + size[1]), fill=fg_color, width=4) ;
			img = img.crop((10, 8 + offset[1], 10 + size[0] + 6, 11 + size[1]))

		elif shape == 'rectangle' :
			draw.rectangle([(10, 8 + offset[1]), (10 + size[0], 11 + size[1])], outline=fg_color) ;
			img = img.crop((7, 5 + offset[1], 10 + size[0] + 3, 11 + size[1] + 3)) ;

	elif symbol != 'none' :
		if symbol == 'bluetooth' or symbol == 'wifi' :
			sym_img = bg_raw_imgs[symbol] ;#Image.open('word_back/' + symbol + '.png') ;
			sym_img = sym_img.resize((15, 7 + size[1])) ;
			img.paste(sym_img, (10 + size[0], 10, 25 + size[0], 17 + size[1])) ;
			img = img.crop((10, 10, 27 + size[0], 20 + size[1])) ;

		elif symbol == 'battery' :
			sym_img = bg_raw_imgs[symbol] ;#Image.open('word_back/' + symbol + '.png') ;
			sym_img = sym_img.resize((30, 5 + size[1])) ;
			img.paste(sym_img, (10 + size[0], 13, 40 + size[0], 18 + size[1])) ;
			img = img.crop((10, 10, 40 + size[0], 20 + size[1])) ;

	else :
		img = img.crop((10, 8 + offset[1], 10 + size[0], 11 + size[1])) ;

	# blend if contains a receipt bg
	if bg not in ['0.png', '29.png'] :
		img = Image.fromarray(tfs.blend(np.array(img) , bg_raw_imgs[bg])) ;

	# apply rotation
	if rotation :	
		img = img.convert("L") ;			
		img = image.random_rotation(np.expand_dims(img, 0), 3)[0] ;
		img = Image.fromarray(img) ;


	# apply elastic distortion
	if elas_dist :
		img = np.float16(img.convert("RGB")) ;
		img = tfs.elastic_transform(img, 20, 2.2) ;

	# apply shadow removal
	#img = img.convert("L") ;
	#img = Image.fromarray(tfs.remove_shadow(np.array(img))) ; 

	# convert to 3 channel
	img = img.convert("RGB") ;

	return img ;


# function to generate word images 
def process(words, start, end, fonts, dest_img, dest_npy, n_trans) :

	global is_gray ;
	font_faces = os.listdir(fonts) ;
	random.shuffle(font_faces)

	# background_files = ['0.png', '29.png', '1.png', '2.png', '3.png', '4.png', '6.png', '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png',
						# '16.png', '17.png', '18.png', '19.png', '20.png','22.png', '23.png', '24.png', '25.png', '26.png', '28.jpg', '30.jpg', '31.jpg']

	background_files = ['0.png', '29.png', '25.png', '26.png', '28.jpg', '31.jpg'] ;

	bg_weights = [ 1.0 / len(background_files) for i in range(len(background_files)) ]
	font_weights = [ 1.0 / len(font_faces) for i in range(len(font_faces)) ]

	bg_len = len(background_files) ;
	font_len = len(font_faces) ;
	idx = 0 ;

	shapes = ['underline', 'rectangle', 'left_line', 'right_line'] ;
	symbols = ['bluetooth', 'wifi', 'battery'] ;

	# reading background images for every process spawned	
	for i in range(len(background_files)) :
		bg_raw_imgs[background_files[i]] = Image.open('word_back/' + background_files[i]).convert("RGBA") ;

	for i in range(len(symbols)) :
		bg_raw_imgs[symbols[i]] = Image.open('word_back/' + symbols[i] + '.png') ;

	# print(bg_raw_imgs) ;


	# sess = tf.Session()	;
	font_i = 0 ;
	imgs = [[] for i in range(n_trans)]
	labels = [[] for i in range(n_trans)]

	for i in range(start,end):

		if i % 250 == 0 :
			print(i , words[i] , 'word length :' , len(words[i])) ;

		# print('Generating', words[i]) ;
		# 'n_trans' transformations for each word
		for j in range(n_trans) :

			try:
				# print('Transformation number', j) ;

				bg = background_files[idx % bg_len] ;
				font_name = font_faces[idx % font_len] ;
				font_size = 80#np.random.randint(low=16, high=80, size=1) ;
				rotation = np.random.choice([0,1]) ;
				elas_dist = 0#np.random.choice([0,1], 1) ;

				# shapes and symbols are mutually exclusive
				if j >= 30 :
					shape = 'none' ;
					symbol = 'none' ;
				elif random.random() > 0.5 :
					shape = np.random.choice(shapes) ;
					symbol = 'none' ;
				else :
					if len(words[i]) > 5 :
						symbol = 'none' ;
					else :
						symbol = np.random.choice(symbols) ;
					shape = 'none' ;


				# print(bg, font_name, font_size, rotation, elas_dist, other) ;

				img = generate_word_image(words[i], bg, font_name, font_size, rotation, elas_dist, shape, symbol) ;
				if img.size[0] < 100 or img.size[1] < 32 :
					continue ;									
				img = img.resize((100,32)) ;

				# Gaussian blur, downsample by half and then upsample by same factor(using bicubic interpolation)
				blurred = cv2.GaussianBlur(np.array(img),(5,5),0) ;
				blurred = cv2.resize(blurred, (50, 16), interpolation=cv2.INTER_CUBIC) ;
				blurred = cv2.resize(blurred, (100, 32), interpolation=cv2.INTER_CUBIC) ;
				blurred = Image.fromarray(blurred) ;

				img_filename = dest_img + str(j) + '/' + str(i) + '.png'		
				blurred_filename = dest_img + str(j) + '/' + str(i) + '_blur.png'		
				
				#img.save(img_filename)
				#blurred.save(blurred_filename)
				
				imgs[j].append(np.float16(blurred)) ;
				labels[j].append(np.float16(img)) ;

				idx += 1 ;
				del img ;
				del blurred ;

			except Exception as e:
				print('Exception ' , j , e, words[i]) ;
				traceback.print_exc(file=sys.stdout)
				continue

		
	# randomly picking 200 images for every transformation
	val_imgs = []#[[] for i in range(30)]
	val_labels = []#[[] for i in range(30)]
	train_imgs = [] 
	train_labels = [] 
	for j in range(n_trans) :

		data,label = imgs[j],labels[j] ;
		val_imgs += data[len(data) - 10 : len(data)] ;
		val_labels += label[len(data) - 10 : len(data)] ;
		train_imgs += data[:len(data) - 10] ;
		train_labels += label[:len(data) - 10] ; 
		

	train_imgs = np.float16(train_imgs) / 255.0 ;
	train_labels = np.float16(train_labels) / 255.0 ;
	val_imgs = np.float16(val_imgs) / 255.0 ;
	val_labels = np.float16(val_labels) / 255.0 ;
	
	
	# saving image/label npys for every process
	np.save(dest_npy + str(start) + '_train.npy',train_imgs) ;
	np.save(dest_npy + str(start) + '_train_labels.npy',train_labels) ;
	np.save(dest_npy + str(start) + '_val.npy',val_imgs) ;
	np.save(dest_npy + str(start) + '_val_labels.npy',val_labels) ;


def main(fonts, words, input_file, dest_path , type, img_type) :

	fid = os.path.splitext(input_file)[0] ;
	dest_img = dest_path + '/' + type + '/' + fid + '/' ;
	dest_npy = dest_path + '/' + type + '/' + fid + '_npys/' ;

	# creating destination image and numpy directories
	if os.path.exists(dest_img) :		
		shutil.rmtree(dest_img) ;
	os.makedirs(dest_img) ;

	if os.path.exists(dest_npy) :
		shutil.rmtree(dest_npy)
	os.makedirs(dest_npy) ;

	n_trans = 40
	for i in range(n_trans):
		path = dest_img + '/' + str(i)
		os.system("mkdir "+path)

	# setting is_gray var
	global is_gray ;
	if img_type == 'gray' :
		is_gray = True ;

	print(len(words))

	nproc = 10 ;
	words_per_process = len(words) // nproc ;
	print('Words per process ' , words_per_process) ;
	proc = [] 
	for i in range(nproc):
		p = Process(target=process,args=(words,words_per_process*i,words_per_process*(i+1),fonts,dest_img,dest_npy,n_trans))
		p.start() ;
		proc.append(p)

	for p in proc :
		p.join() ;
