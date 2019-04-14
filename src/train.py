import logging
import os
import shutil
from collections import OrderedDict

import torch
from math import log10
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange

import config
from Dataset.LRHR_Dataset import DatasetFromFolder
from Models.architecture import *
from Models.loss import GANLoss
from Models.networks import init_weights
from utils.data_utils import display_transform
from utils.pytorch_ssim import ssim

logger = logging.getLogger('base')

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_set = DatasetFromFolder(mode = 'train')
	valid_set = DatasetFromFolder(mode = 'valid')
	train_loader = DataLoader(dataset = train_set, num_workers = 4, batch_size = config.BATCH_SIZE, shuffle = True,
							  pin_memory = False, drop_last = True)
	val_loader = DataLoader(dataset = valid_set, num_workers = 2, batch_size = 1, shuffle = False, drop_last = True)

	try:
		shutil.rmtree('./output/Statistics/logs')
	except Exception:
		pass
	os.mkdir('./output/Statistics/logs')
	writer = SummaryWriter(log_dir = './output/Statistics/logs')

	netG = RRDBNet(in_nc = 3, out_nc = 3, nf = 16, nb = 16, gc = 32, upscale = config.UPSCALE_FACTOR).to(device)
	print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
	netD = Discriminator_VGG_224(in_nc = 3, base_nf = 16).to(device)
	print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

	# optimizers
	optimizers = list()
	optim_params = []
	for k, v in netG.named_parameters():  # can optimize for a part of the model
		if v.requires_grad:
			optim_params.append(v)

	optimizer_G = torch.optim.Adam(optim_params, lr = config.lr_g, weight_decay = 0, betas = (0.9, 0.999))
	optimizer_D = torch.optim.Adam(netD.parameters(), lr = config.lr_d, weight_decay = 0, betas = (0.9, 0.999))


	if config.WARM_START:
		checkpointD = torch.load(f"output/models/netD_epoch={config.N_EPOCHS_START}")
		netD.load_state_dict(checkpointD['model_state_dict'])
		optimizer_D.load_state_dict(checkpointD['optimizer_state_dict'])

		checkpointG = torch.load(f"output/models/netG_epoch={config.N_EPOCHS_START}")
		netG.load_state_dict(checkpointG['model_state_dict'])
		optimizer_G.load_state_dict(checkpointG['optimizer_state_dict'])
	else:
		init_weights(netG, init_type = 'kaiming', scale = 0.1)
		init_weights(netD, init_type = 'kaiming', scale = 1)

	optimizers.append(optimizer_G)
	optimizers.append(optimizer_D)

	# G pixel loss
	cri_pix = nn.L1Loss().to(device)
	# G feature loss
	cri_fea = nn.L1Loss().to(device)
	# load VGG perceptual loss
	netF = VGGFeatureExtractor(feature_layer = 34, use_bn = False).to(device)
	print('# perceptual parameters:', sum(param.numel() for param in netF.parameters()))
	# GD gan loss
	cri_gan = GANLoss("vanilla", 1.0, 0.0).to(device)

	# schedulers
	schedulers = list()
	for optimizer in optimizers:
		schedulers.append(lr_scheduler.MultiStepLR(optimizer, [50000, 100000, 200000, 300000], 0.5))

	log_dict = OrderedDict()

	if config.MODE == 'train':
		netD.train()
		netG.train()
		global_step = 0
		for epoch in trange(config.N_EPOCHS_START, config.N_EPOCHS_END):
			train_bar = tqdm(train_loader)
			train_bar.set_description_str(desc = f"N epochs - {epoch}")

			for scheduler in schedulers:
				scheduler.step()

			for step, (lr, hr) in enumerate(train_bar):
				global_step += 1
				############################
				# (1) Update D network
				###########################
				for p in netD.parameters():
					p.requires_grad = True

				optimizer_D.zero_grad()

				lr = torch.autograd.Variable(lr, requires_grad = True).to(device)
				hr = torch.autograd.Variable(hr, requires_grad = True).to(device)

				sr = netG(lr)

				pred_d_real = netD(hr)
				pred_d_fake = netD(sr.detach())

				l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
				l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
				l_d_total = (l_d_real + l_d_fake) / 2
				l_d_total.backward()
				optimizer_D.step()

				############################
				# (2) Update G network
				###########################
				if step % config.D_update_ratio == 0 and step > config.D_init_iters:
					for p in netD.parameters():
						p.requires_grad = False
					optimizer_G.zero_grad()

					l_g_total = 0
					# pixel loss
					l_g_pix = config.l_pix_w * cri_pix(sr, hr)
					l_g_total += l_g_pix
					# feature loss
					real_fea = netF(hr).detach()
					fake_fea = netF(sr)
					l_g_fea = config.l_fea_w * cri_fea(fake_fea, real_fea)
					l_g_total += l_g_fea
					# G gan + cls loss
					pred_g_fake = netD(sr)
					pred_d_real = netD(hr).detach()

					l_g_gan = config.l_gan_w * (cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
												cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
					l_g_total += l_g_gan

					l_g_total.backward()
					optimizer_G.step()

				# set log
				if step % config.D_update_ratio == 0 and step > config.D_init_iters:
					# G
					log_dict['l_g_pix'] = l_g_pix.item()
					log_dict['l_g_fea'] = l_g_fea.item()
					log_dict['l_g_gan'] = l_g_gan.item()
				# D
				log_dict['l_d_real'] = l_d_real.item()
				log_dict['l_d_fake'] = l_d_fake.item()

				# D outputs
				log_dict['D_real'] = torch.mean(pred_d_real.detach())
				log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

				writer.add_scalar(tag = "D_real", scalar_value = log_dict['D_real'], global_step = global_step)
				writer.add_scalar(tag = "D_fake", scalar_value = log_dict['D_fake'], global_step = global_step)

				# validation
				if global_step % config.val_freq == config.val_freq - 1:
					netG.eval()
					out_path = 'output/training_results/SRF_' + str(config.UPSCALE_FACTOR) + '/'
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

						if val_step < 5:
							val_images.extend(
								[display_transform()(nn.functional.interpolate(lr[0].cpu(), scale_factor = config.UPSCALE_FACTOR)),
								 display_transform()(hr.data.cpu().squeeze(0)),
								 display_transform()(sr.data.cpu().squeeze(0))])
						if val_step > 1000:
							break

					valing_results['mse'] = valing_results['mse'] / val_loader.__len__()
					valing_results['psnr'] = 10 * log10(1 / valing_results['mse'])
					valing_results['ssim'] = valing_results['ssims'] / val_loader.__len__()

					val_images = torch.stack(val_images)
					val_images = torch.chunk(val_images, val_images.size(0) // 3)

					for index, image in enumerate(val_images, 1):
						image = make_grid(image, nrow = 3, padding = 5)
						save_image(image, out_path + 'global_step_%d_index_%d.png' % (global_step, index), padding = 5)
					# log
					writer.add_scalar(tag = "mse", scalar_value = valing_results['mse'], global_step = global_step)
					writer.add_scalar(tag = "psnr", scalar_value = valing_results['psnr'], global_step = global_step)
					writer.add_scalar(tag = "ssim", scalar_value = valing_results['ssim'], global_step = global_step)

				# save models and training states
				if global_step % config.save_checkpoint_freq == config.save_checkpoint_freq - 1:
					logger.info('Saving models and training states.')
					torch.save({'epoch': epoch,
								'model_state_dict': netG.state_dict(),
								'optimizer_state_dict': optimizer_G.state_dict()},
							   f'output/models/netG_step={global_step}')
					torch.save({'epoch': epoch,
								'model_state_dict': netD.state_dict(),
								'optimizer_state_dict': optimizer_D.state_dict()},
							   f'output/models/netD_step={global_step}')

			logger.info('Saving the final model.')
			torch.save({'epoch': epoch,
						'model_state_dict': netG.state_dict(),
						'optimizer_state_dict': optimizer_G.state_dict()},
					    f'output/models/netG_epoch={epoch}')
			torch.save({'epoch': epoch,
						'model_state_dict': netD.state_dict(),
						'optimizer_state_dict': optimizer_D.state_dict()},
					    f'output/models/netD_epoch={epoch}')
			logger.info('End of training.')
