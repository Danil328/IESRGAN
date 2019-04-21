import os
import os.path
import sys
from multiprocessing import Pool
import cv2
import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.util import imresize_np
from utils.progress_bar import ProgressBar
import config

@click.command()
@click.option('--input_folder', default='/media/danil/Data/Experiments/ESRGAN/data/DIV2K_train_HR_sub', help='path to data')
@click.option('--save_folder', default='/media/danil/Data/Experiments/ESRGAN/data/DIV2K_train_HR_sub_bicLRx', help='path to save data')
def main(input_folder, save_folder):
	"""A multi-thread tool to crop sub imags."""
	n_thread = 8
	compression_level = 0  # 3 is the default value in cv2
	# CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
	# compression time. If read raw images during training, use 0 for faster IO speed.
	save_folder += str(config.UPSCALE_FACTOR)
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
		print('mkdir [{:s}] ...'.format(save_folder))
	else:
		print('Folder [{:s}] already exists. Exit...'.format(save_folder))


	img_list = []
	for root, _, file_list in sorted(os.walk(input_folder)):
		path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
		img_list.extend(path)

	def update(arg):
		pbar.update(arg)

	pbar = ProgressBar(len(img_list))

	print("Start 2 ...")
	pool = Pool(n_thread)
	for path in img_list:
		pool.apply_async(worker,
						 args=(path, save_folder, compression_level),
						 callback=update)
	pool.close()
	pool.join()
	print('All subprocesses done.')


def worker(path, save_folder, compression_level):
	img_name = os.path.basename(path)
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED) / 255.0
	img_lr = imresize_np(img, scale=1.0 / config.UPSCALE_FACTOR, antialiasing=True)
	cv2.imwrite(
		os.path.join(save_folder, img_name),
		(img_lr*255).astype(int), [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
	return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
	main()
