#!/usr/bin/env bash

python Dataset/extract_sibing_single.py --input_folder /media/danil/Data/Datasets/DIV2K/DIV2K_valid_HR/ --save_folder /media/danil/Data/Experiments/ESRGAN/data/DIV2K_valid_HR_sub/
python Dataset/generate_LR_images.py --input_folder /media/danil/Data/Experiments/ESRGAN/data/DIV2K_valid_HR_sub --save_folder /media/danil/Data/Experiments/ESRGAN/data/DIV2K_valid_HR_sub_bicLRx4