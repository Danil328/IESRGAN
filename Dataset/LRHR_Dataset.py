import glob

import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor

import config


class DatasetFromFolder(data.Dataset):
    def __init__(self, mode):
        super(DatasetFromFolder, self).__init__()
        if mode == 'train':
            self.hr_images = glob.glob(config.PATH_TO_TRAIN_HR_DATA + '/*.png')
            self.lr_images = glob.glob(config.PATH_TO_TRAIN_LR_DATA + '/*.png')
        else:
            self.hr_images = glob.glob(config.PATH_TO_VALID_HR_DATA + '/*.png')
            self.lr_images = glob.glob(config.PATH_TO_VALID_LR_DATA + '/*.png')
        assert len(self.hr_images) == len(self.lr_images), 'Count HR images must be equal count LR images!'

    def __getitem__(self, index):
        hr_image = ToTensor()(Image.open(self.hr_images[index]))
        lr_image = ToTensor()(Image.open(self.lr_images[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_images)
