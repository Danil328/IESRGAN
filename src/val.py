from math import log10
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from Dataset.LRHR_Dataset import DatasetFromFolder
from models.architecture import *
from utils.pytorch_ssim import ssim

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	valid_set = DatasetFromFolder(mode = 'valid')
	val_loader = DataLoader(dataset = valid_set, num_workers = 2, batch_size = 1, shuffle = False, drop_last = True)

	netG = RRDBNet(in_nc = 3, out_nc = 3, nf = 16, nb = 16, gc = 32, upscale = config.UPSCALE_FACTOR)
	netG.load_state_dict(torch.load(f"output/models/netG_epoch={10}")['model_state_dict'])
	netG.to(device).eval()

	val_bar = tqdm(val_loader)

	valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0}
	val_images = []
	for val_step, (lr, hr) in enumerate(val_bar):
		lr = lr.to(device)
		hr = hr.to(device)

		sr = netG(lr)
		valing_results['mse'] += ((sr - hr) ** 2).data.mean()
		valing_results['ssims'] += ssim(sr, hr).data.item()

	valing_results['mse'] = valing_results['mse'] / val_loader.__len__()
	valing_results['psnr'] = 10 * log10(1 / valing_results['mse'])
	valing_results['ssim'] = valing_results['ssims'] / val_loader.__len__()

	print(valing_results)
