import cv2
from math import log10
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import config
from dataset.LRHR_Dataset import DatasetFromFolder
from models.architecture import *
from utils.pytorch_ssim import ssim
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	netG = RRDBNet(in_nc = 3, out_nc = 3, nf = 16, nb = 16, gc = 32, upscale = config.UPSCALE_FACTOR)
	netG.load_state_dict(torch.load(f"output/models/netG_epoch={10}")['model_state_dict'])
	netG.to(device).eval()

	path_to_image = ''
	img = Image.open(path_to_image)

	with torch.no_grad():
		img_tensor = ToTensor()(Image.open(img)).unsqueeze(0).to(device)
		sr = netG(img_tensor)

	sr_pil = Image.fromarray(sr.data.cpu().numpy())

	# read the image and define the stepSize and window size
	# (width,height)
	image = cv2.imread("cell.png")  # your image path
	tmp = image  # for drawing a rectangle
	stepSize = 50
	(w_width, w_height) = (50, 50)  # window size
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

