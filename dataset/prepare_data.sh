#!/usr/bin/env bash

python extract_sibing_single.py --input_folder ../data/DIV2K_train_HR --save_folder ../data/train_HR
python extract_sibing_single.py --input_folder ../data/Flickr2K --save_folder ../data/train_HR
python extract_sibing_single.py --input_folder ../data/DIV2K_valid_HR/ --save_folder ../data/valid_HR

python generate_LR_images.py --input_folder ../data/train_HR --save_folder ../data/train_LRx
python generate_LR_images.py --input_folder ../data/valid_HR --save_folder ../data/valid_LRx
