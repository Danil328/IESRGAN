import glob
import random

import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image


class DatasetFromFolder(data.Dataset):
    def __init__(self, mode, config):

        super(DatasetFromFolder, self).__init__()
        if mode == 'train':
            self.hr_images = glob.glob(config['PATH_TO_TRAIN_HR_DATA'] + '/*.png')
            self.lr_images = glob.glob(config['PATH_TO_TRAIN_LR_DATA'] + '/*.png')
        else:
            self.hr_images = glob.glob(config['PATH_TO_VALID_HR_DATA'] + '/*.png')
            self.lr_images = glob.glob(config['PATH_TO_VALID_LR_DATA'] + '/*.png')

        assert len(self.hr_images) == len(self.lr_images), 'Count HR images must be equal count LR images!'
        assert list(map(lambda x: x.split('/')[-1], self.hr_images)) == list(map(lambda x: x.split('/')[-1], self.hr_images)), 'List HR images must be equal List LR images!'

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_images[index])
        lr_image = Image.open(self.lr_images[index])
        lr_image, hr_image = self.transform(lr_image, hr_image)
        return lr_image, hr_image

    @staticmethod
    def transform(image, mask):
        # Random horizontal flipping
        if random.random() > 0.5:

            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.hr_images)
