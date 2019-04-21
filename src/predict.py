import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from models.architecture import *

if __name__ == '__main__':
	with open('config.json', 'r') as f:
		config = json.load(f)

	default_config = config['DEFAULT']
	config = config['PREDICT']

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	netG = RRDBNet(in_nc = 3, out_nc = 3, nf = 16, nb = 16, gc = 32, upscale = default_config['upscale_factor'])
	netG.load_state_dict(torch.load(config['path_to_model'])['model_state_dict'])
	netG.to(device).eval()

	img = Image.open(config['path_to_image'])

	with torch.no_grad():
		img_tensor = ToTensor()(Image.open(img)).unsqueeze(0).to(device)
		sr = netG(img_tensor)

	sr_pil = Image.fromarray(sr.data.cpu().numpy())

	# read the image and define the stepSize and window size
	# (width,height)
	image = cv2.imread("cell.png")  # your image path
	tmp = image  # for drawing a rectangle
	stepSize = default_config['crop_size']
	(w_width, w_height) = (default_config['crop_size'], default_config['crop_size'])  # window size
	for x in range(0, image.shape[1] - w_width, stepSize):
		for y in range(0, image.shape[0] - w_height, stepSize):
			window = image[x:x + w_width, y:y + w_height, :]

			# classify content of the window with your classifier and
			# determine if the window includes an object (cell) or not
			# draw window on image
			cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
			plt.imshow(np.array(tmp).astype('uint8'))
	# show all windows
	plt.show()

