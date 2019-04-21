UPSCALE_FACTOR = 2

PATH_TO_TRAIN_LR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/Flickr2K_train_HR_sub_bicLRx{}'.format(UPSCALE_FACTOR)
PATH_TO_TRAIN_HR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/Flickr2K_train_HR_sub'

PATH_TO_VALID_HR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/Flickr2K_valid_HR_sub'
PATH_TO_VALID_LR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/Flickr2K_valid_HR_sub_bicLRx{}'.format(UPSCALE_FACTOR)

MODE = 'train'
pretrain_generator = True

# TRAINING PARAMETERS
N_EPOCHS_START = 1
N_EPOCHS_END = 2
BATCH_SIZE = 3
lr_g = 1e-4
lr_d = 1e-4
WARM_START = False

D_update_ratio = 2
D_init_iters = 10
val_freq = 5e3
save_checkpoint_freq = 5e3

#loss
# G pixel loss
l_pix_w = 1e-2
# G feature loss
l_fea_w = 1
# GD gan loss
l_gan_w = 1e-2