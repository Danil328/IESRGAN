#!/usr/bin/env bash

python Dataset/extract_sibing_single.py --input_folder /media/danil/Data/Datasets/Flickr2K/train/ --save_folder /media/danil/Data/Experiments/ESRGAN/data/Flickr2K_train_HR_sub/
python Dataset/generate_LR_images.py --input_folder /media/danil/Data/Experiments/ESRGAN/data/Flickr2K_train_HR_sub --save_folder /media/danil/Data/Experiments/ESRGAN/data/Flickr2K_train_HR_sub_bicLRx

python Dataset/extract_sibing_single.py --input_folder /media/danil/Data/Datasets/Flickr2K/val/ --save_folder /media/danil/Data/Experiments/ESRGAN/data/Flickr2K_valid_HR_sub/
python Dataset/generate_LR_images.py --input_folder /media/danil/Data/Experiments/ESRGAN/data/Flickr2K_valid_HR_sub --save_folder /media/danil/Data/Experiments/ESRGAN/data/Flickr2K_valid_HR_sub_bicLRx