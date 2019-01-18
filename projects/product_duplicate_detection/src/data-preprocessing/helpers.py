import requests
import numpy as np
import cv2
import scipy.ndimage as ndi

# method to download image from a given url
def download_from_url(url):
	data = []
	url = url.strip()
	try:
		r = requests.get(url)#, timeout = 1)
		data = r.content
	except:
		print('Error {} downloading url {}'.format(r.status_code, url))
	return data;

def write_image(data, path):
	with open(path, 'wb') as f:
		f.write(data);

def read_image(path):
	img = cv2.imread(path);
	if img is not None:
		return img;


# methods to rotate an rgb image
def transform_matrix_offset_center(matrix, x, y):
  o_x = float(x) / 2 + 0.5
  o_y = float(y) / 2 + 0.5
  offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
  reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
  transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
  return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
  x = np.rollaxis(x, channel_axis, 0)
  final_affine_matrix = transform_matrix[:2, :2]
  final_offset = transform_matrix[:2, 2]
  channel_images = [
      ndi.interpolation.affine_transform(
          x_channel,
          final_affine_matrix,
          final_offset,
          order=1,
          mode=fill_mode,
          cval=cval) for x_channel in x
  ]
  x = np.stack(channel_images, axis=0)
  x = np.rollaxis(x, 0, channel_axis + 1)
  return x


def random_rotation(x,
                    rg,
                    row_axis=1,
                    col_axis=2,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):

  deg = np.random.uniform(-rg, rg) ;
  theta = np.deg2rad(deg)
  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

  h, w = x.shape[0], x.shape[1]
  transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

  # rotating every channel
  ch1 = apply_transform(np.expand_dims(x[:,:,0], 0), transform_matrix, channel_axis, fill_mode, cval)[0]
  ch2 = apply_transform(np.expand_dims(x[:,:,1], 0), transform_matrix, channel_axis, fill_mode, cval)[0]
  ch3 = apply_transform(np.expand_dims(x[:,:,2], 0), transform_matrix, channel_axis, fill_mode, cval)[0]

  x = np.dstack((ch1, ch2, ch3)) ;
  return x
