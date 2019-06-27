import json
import sys
sys.path.append("..")
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import trange, tqdm
from models.srgan import Generator as SRGAN_G
from sklearn.preprocessing import MinMaxScaler
from models.architecture import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

if __name__ == '__main__':
    with open('../config.json', 'r') as f:
        config = json.load(f)

    default_config = config['DEFAULT']
    config = config['PREDICT']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = SRGAN_G(default_config['upscale_factor'])
    netG.load_state_dict(torch.load(config['path_to_model'])['model_state_dict'])
    netG.to(device).eval()

    scaler = MinMaxScaler(feature_range=(0, 1))

    image_paths = glob.glob(config['path_to_image_folder'])
    for im_path in tqdm(image_paths):
        img = Image.open(im_path)
        # if img.size[1] < 400 and img.size[0] < 400:
        #     img = img.resize((int(img.size[0]*2), int(img.size[1]*2)))
        if img.mode == 'L':
            img = img.convert('RGB')

        with torch.no_grad():
            img_tensor = ToTensor()(img).unsqueeze(0).to(device)
            sr = netG(img_tensor)


        sr = sr.data.squeeze().cpu().numpy()
        sr[sr < 0] = 0
        sr[sr > 1] = 1
        sr = np.moveaxis(sr * 255, 0, -1).astype(np.uint8)

        # img_yuv = cv2.cvtColor(sr, cv2.COLOR_BGR2YUV)
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # sr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        #sr = cv2.resize(sr, (int(sr.shape[1]/2), int(sr.shape[0]/2)))

        # kernel = np.ones((3, 3), np.float32) / 9
        # sr = cv2.filter2D(sr, -1, kernel)
        # sr = cv2.blur(sr, (3, 3))
        sr = cv2.fastNlMeansDenoisingColored(sr, None, 5, 5, 5, 15)

        cv2.imwrite(im_path.replace('images', 'sr_images_SRGAN'), sr[..., ::-1])

        # scale = default_config['upscale_factor']
        # image = np.asarray(img)
        # sr_image = np.zeros((image.shape[0] * scale, image.shape[1] * scale, image.shape[2]), dtype=np.uint8)
        # stepSize = default_config['crop_size']
        # (w_width, w_height) = (default_config['crop_size'], default_config['crop_size'])  # window size
        # for x in range(0, image.shape[0], stepSize):
        #     for y in range(0, image.shape[1], stepSize):
        #         window = image[x:x + w_width, y:y + w_height, :]
        #         window_tensor = ToTensor()(window).unsqueeze(0).to(device)
        #         window_sr = netG(window_tensor)
        #         window_sr[window_sr < 0] = 0
        #         window_sr[window_sr > 1] = 1
        #         sr_image[x * scale:x * scale + window.shape[0] * scale, y * scale:y * scale + window.shape[1] * scale, :] = np.moveaxis(window_sr.data.squeeze().cpu().numpy() * 255, 0, -1).astype(np.uint8)
        # # show all windows
        # # plt.show()
        # # plt.figure()
        # # plt.imshow(sr_image)
        # # plt.show()
        # # cv2.imwrite(config['path_to_save_image', sr_image[..., ::-1]])
        # cv2.imwrite(im_path.replace('images', 'sr_images_SRGAN_window'), sr_image[..., ::-1])
