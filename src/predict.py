import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
#from skimage.io import imshow, imread, imsave
from tqdm import trange

from models.architecture import *

if __name__ == '__main__':
	with open('../config.json', 'r') as f:
		config = json.load(f)

	default_config = config['DEFAULT']
	config = config['PREDICT']

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	netG = RRDBNet(in_nc = 3, out_nc = 3, nf = 16, nb = 16, gc = 32, upscale = default_config['upscale_factor'])
	netG.load_state_dict(torch.load(config['path_to_model'])['model_state_dict'])
	netG.to(device).eval()

	# img = Image.open(config['path_to_image'])
	img = Image.open('../images/Danil.jpeg')

	# with torch.no_grad():
	# 	img_tensor = ToTensor()(img).unsqueeze(0).to(device)
	# 	sr = netG(img_tensor)
	#
	# sr[sr < 0] = 0
	# sr[sr > 1] = 1
	# sr = np.moveaxis(sr.data.squeeze().cpu().numpy()*255, 0, -1).astype(int)
	# cv2.imwrite("123.png", sr[..., ::-1])


	scale = default_config['upscale_factor']
	image = np.asarray(img)
	sr_image = np.zeros((image.shape[0]*scale, image.shape[1]*scale, image.shape[2]), dtype = np.uint8)
	tmp = image.copy()  # for drawing a rectangle
	stepSize = default_config['crop_size']
	(w_width, w_height) = (default_config['crop_size'], default_config['crop_size'])  # window size
	for x in range(0, image.shape[0], stepSize):
		for y in range(0, image.shape[1], stepSize):
			window = image[x:x + w_width, y:y + w_height, :]
			window_tensor = ToTensor()(window).unsqueeze(0).to(device)
			window_sr = netG(window_tensor)
			window_sr[window_sr < 0] = 0
			window_sr[window_sr > 1] = 1
			sr_image[x*scale:x*scale + window.shape[0]*scale, y*scale:y*scale + window.shape[1]*scale, :] = np.moveaxis(window_sr.data.squeeze().cpu().numpy()*255, 0, -1).astype(np.uint8)
	# show all windows
	plt.show()
	plt.figure()
	plt.imshow(sr_image)
	plt.show()
	# cv2.imwrite(config['path_to_save_image', sr_image[..., ::-1]])
	cv2.imwrite("../images/DanilSR4.png", sr_image[..., ::-1])

