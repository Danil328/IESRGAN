import glob
import json
import random
import cv2
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from albumentations import (
    Blur, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, OneOf, Compose
)
import numpy as np

class DatasetFromFolder(data.Dataset):
    def __init__(self, mode, config):

        super(DatasetFromFolder, self).__init__()
        if mode == 'train':
            self.hr_images = glob.glob(config['PATH_TO_TRAIN_HR_DATA'] + '/*.png')
            self.lr_images = glob.glob(config['PATH_TO_TRAIN_LR_DATA'] + '/*.png')
        else:
            self.hr_images = glob.glob(config['PATH_TO_VALID_HR_DATA'] + '/*.png')
            self.lr_images = glob.glob(config['PATH_TO_VALID_LR_DATA'] + '/*.png')

        self.aug = self.augment_lr_image(p=0.5)

        assert len(self.hr_images) == len(self.lr_images), 'Count HR images must be equal count LR images!'
        assert list(map(lambda x: x.split('/')[-1], self.hr_images)) == list(
            map(lambda x: x.split('/')[-1], self.hr_images)), 'List HR images must be equal List LR images!'

        self.lr_images = list(map(lambda x: x.replace('HR4', 'LRx4'), self.hr_images))

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_images[index])
        lr_image = Image.open(self.lr_images[index])
        lr_image, hr_image = self.transform(lr_image, hr_image)
        return lr_image, hr_image


    def transform(self, lr_image, hr_image):
        # Random horizontal flipping
        if random.random() > 0.5:
            lr_image = TF.hflip(lr_image)
            hr_image = TF.hflip(hr_image)

        # Random vertical flipping
        if random.random() > 0.5:
            lr_image = TF.vflip(lr_image)
            hr_image = TF.vflip(hr_image)

        # Apply AUG to LR image
        lr_image = self.aug(image=np.asarray(lr_image))['image']

        # Transform to tensor
        lr_image = TF.to_tensor(Image.fromarray(lr_image))
        hr_image = TF.to_tensor(hr_image)
        return lr_image, hr_image

    @staticmethod
    def augment_lr_image(p=.5):
        return Compose([
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.25),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.25)
        ], p=p)

    def __len__(self):
        return len(self.hr_images)


if __name__ == '__main__':
    with open('../config.json', 'r') as f:
        config = json.load(f)
    default_config = config['DEFAULT']

    dataset = DatasetFromFolder(mode='train', config=default_config)
    lr_image, hr_image = dataset.__getitem__(3) #np.random.randint(0, 100, 1)[0]
    print(lr_image.min(), lr_image.max())
    print(hr_image.min(), hr_image.max())
    lr_image = np.moveaxis(lr_image.numpy(), 0, -1)
    upscale_lr = (cv2.resize(lr_image, dsize=(default_config['crop_size'], default_config['crop_size']))*255).astype(int)
    hr_image = (np.moveaxis(hr_image.numpy(), 0, -1)*255).astype(int)

    cv2.imwrite("lr_image.png", upscale_lr)
    cv2.imwrite("hr_image.png", hr_image)
