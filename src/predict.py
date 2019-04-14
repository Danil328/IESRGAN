from math import log10
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import config
from Dataset.LRHR_Dataset import DatasetFromFolder
from models.architecture import *
from utils.pytorch_ssim import ssim
from torchvision.transforms import ToTensor



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

