PATH_TO_TRAIN_LR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/DIV2K_train_HR_sub_bicLRx4'
PATH_TO_TRAIN_HR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/DIV2K_train_HR_sub'

PATH_TO_VALID_HR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/DIV2K_valid_HR_sub'
PATH_TO_VALID_LR_DATA = '/media/danil/Data/Experiments/ESRGAN/data/DIV2K_valid_HR_sub_bicLRx4'

MODE = 'train'
UPSCALE_FACTOR = 4

# TRAINING PARAMETERS
N_EPOCHS = 10
BATCH_SIZE = 2
lr_g = 1e-4
lr_d = 1e-4
WARM_START = False
BATCH_ACCUMALATION = True
N_BATCH_UPDATE = 10
D_update_ratio = 1
D_init_iters = 5
val_freq = 5e3
save_checkpoint_freq = 5e3