import sys

sys.path.append('..')
import logging
import os
import shutil
from collections import OrderedDict

from math import log10
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange

from dataset.LRHR_Dataset import DatasetFromFolder
from models.architecture import *
from models.networks import init_weights
from utils.data_utils import display_transform
from utils.pytorch_ssim import ssim
from models.srgan import Generator as SRGAN_G
import json

logger = logging.getLogger('base')

if __name__ == '__main__':

    with open('../config.json', 'r') as f:
        config = json.load(f)

    default_config = config['DEFAULT']
    config = config['TRAIN_PSNR']

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_set = DatasetFromFolder(mode='train', config=default_config)
    valid_set = DatasetFromFolder(mode='valid', config=default_config)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=config['batch_size'], shuffle=True,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(dataset=valid_set, num_workers=4, batch_size=1, shuffle=False,
                            drop_last=False)

    # TensorBoard
    try:
        shutil.rmtree(config['log_path'])
    except Exception:
        pass
    os.mkdir(config['log_path'])
    writer = SummaryWriter(log_dir=config['log_path'])

    input_shape = int(default_config['crop_size'] / default_config['upscale_factor'])
    output_shape = int(default_config['crop_size'])

    # Define Model
    # netG = RRDBNet(in_nc=3, out_nc=3, nf=16, nb=16, gc=32, upscale=default_config['upscale_factor']).to(device)
    netG = SRGAN_G(default_config['upscale_factor']).to(device)
    netF = VGGFeatureExtractor(feature_layer=34, use_bn=False).to(device)

    if len(config['path_to_G_model']) > 0:
        state = torch.load(config['path_to_G_model'], map_location=torch.device('cpu'))
        layers = list(state.keys())
        bn_layers = [i for i in layers if 'bn' in i]
        for i in bn_layers:
            del state[i]
        netG.load_state_dict(state, strict=False)
    else:
        init_weights(netG, init_type='normal', scale=0.1)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (config['ngpu'] > 1):
        netG = nn.DataParallel(netG, [2, 3])
        netF = nn.DataParallel(netF, [2, 3])

    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# perceptual parameters:', sum(param.numel() for param in netF.parameters()))

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=config['learning_rate'], weight_decay=0, betas=(0.9, 0.999))
    global_step = 0

    # G pixel loss
    cri_pix = nn.L1Loss().to(device)
    # G feature loss
    cri_fea = nn.L1Loss().to(device)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.8)

    log_dict = OrderedDict()
    for epoch in trange(config['number_epochs']):
        netG.train()
        train_bar = tqdm(train_loader)
        train_bar.set_description_str(desc=f"N epochs - {epoch}")

        scheduler.step()

        for step, (lr, hr) in enumerate(train_bar):
            global_step += 1

            lr = torch.autograd.Variable(lr, requires_grad=True).to(device)
            hr = torch.autograd.Variable(hr, requires_grad=True).to(device)
            sr = netG(lr)

            optimizer_G.zero_grad()

            l_g_total = 0
            # pixel loss
            l_g_pix = config['loss_pix_weight'] * cri_pix(sr, hr)
            l_g_total += l_g_pix
            # feature loss
            real_fea = netF(hr).detach()
            fake_fea = netF(sr)
            l_g_fea = config['loss_feature_weight'] * cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea

            l_g_total.backward()
            optimizer_G.step()

            log_dict['l_g_pix'] = l_g_pix.item()
            log_dict['l_g_fea'] = l_g_fea.item()

            writer.add_scalar(tag="l_g_pix", scalar_value=log_dict['l_g_pix'], global_step=global_step)
            writer.add_scalar(tag="l_g_fea", scalar_value=log_dict['l_g_fea'], global_step=global_step)

            # validation
            if global_step % config['val_freq'] == config['val_freq'] - 1:
                netG.eval()
                out_path = config['path_to_training_results'] + '/SRF_' + str(default_config['upscale_factor']) + '/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0}
                val_images = []
                for val_step, (lr, hr) in enumerate(val_bar):
                    lr = lr.to(device)
                    hr = hr.to(device)

                    sr = netG(lr)
                    valing_results['mse'] += ((sr - hr) ** 2).data.mean()
                    valing_results['ssims'] += ssim(sr, hr).data.item()

                    if val_step < 10:
                        val_images.extend(
                            [display_transform()(lr[0].cpu()),
                             display_transform()(hr.data.cpu().squeeze(0)),
                             display_transform()(sr.data.cpu().squeeze(0))])
                    else:
                        break

                valing_results['mse'] = valing_results['mse'] / val_loader.__len__()
                valing_results['psnr'] = 10 * log10(1 / valing_results['mse'])
                valing_results['ssim'] = valing_results['ssims'] / val_loader.__len__()

                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 3)

                for index, image in enumerate(val_images, 1):
                    image = make_grid(image, nrow=3, padding=5, normalize=True)
                    save_image(image, out_path + 'global_step_%d_index_%d.png' % (global_step, index), padding=5)
                # log
                writer.add_scalar(tag="mse", scalar_value=valing_results['mse'], global_step=global_step)
                writer.add_scalar(tag="psnr", scalar_value=valing_results['psnr'], global_step=global_step)
                writer.add_scalar(tag="ssim", scalar_value=valing_results['ssim'], global_step=global_step)

                netG.train()

            # save models and training states
            if global_step % config['save_checkpoint_freq'] == config['save_checkpoint_freq'] - 1:
                logger.info('Saving models and training states.')
                torch.save({'epoch': epoch,
                            'model_state_dict': netG.module.state_dict(),
                            'optimizer_state_dict': optimizer_G.state_dict()},
                           f'{config["path_to_save_model"]}/netG_step={global_step}')

        logger.info('Saving the final model.')
        torch.save({'epoch': epoch,
                    'model_state_dict': netG.module.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict()},
                   f'{config["path_to_save_model"]}/netG_epoch={epoch}')
