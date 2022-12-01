from torch.utils.data import Dataset
from PIL import Image
from pixel2style2pixel.utils import data_utils
import torch
import numpy as np
import glob
import os
import random
from masking_utils import mask_upsample
import pandas as pd

from pixel2style2pixel.utils.common import tensor2im
from argparse import Namespace
from models.psp import pSp


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im


class ToOneHot(object):
	""" Convert the input PIL image to a one-hot torch tensor """
	def __init__(self, n_classes=None):
		self.n_classes = n_classes

	def __call__(self, img):
		img = torch.Tensor(np.array(img))
		one_hot = torch.nn.functional.one_hot(img.type(torch.LongTensor), self.n_classes)
		return one_hot.permute(0, 3, 1, 2)


class RGBSegMaskDataset(Dataset):
	def __init__(self, db_path, image_dir_prefix, mask_dir_prefix, source_transform, target_transform, resolution=256):
		self.db_path = db_path
		self.image_dir_prefix = image_dir_prefix
		self.mask_dir_prefix = mask_dir_prefix
		self.mask_fnames = list(glob.glob(os.path.join(self.db_path, self.mask_dir_prefix, '*/*.png')))
		self.source_transform = source_transform
		self.target_transform = target_transform
		print(len(self.mask_fnames))

	def __len__(self):
		return len(self.mask_fnames)

	def mask_path_to_image_path(self, mask_path):
		filename = mask_path.split('/')[-1]
		file_num, file_ext = filename.split('.')
		return os.path.join(self.db_path, self.image_dir_prefix, str(int(file_num)) + '.jpg')

	def __getitem__(self, idx):
		mask_fname = self.mask_fnames[idx]
		image_fname = self.mask_path_to_image_path(mask_path=mask_fname)
		rgb_image = self.target_transform(Image.open(image_fname))
		mask_image = self.source_transform(Image.open(mask_fname).convert('L'))
		return mask_image, rgb_image


class RGBSuperResDataset(Dataset):
	def __init__(self, db_path, image_dir_prefix, mask_dir_prefix, source_transform, target_transform, resolution=256):
		self.db_path = db_path
		self.image_dir_prefix = image_dir_prefix
		self.mask_dir_prefix = mask_dir_prefix
		self.mask_fnames = list(glob.glob(os.path.join(self.db_path, self.mask_dir_prefix, '*/*.png')))
		self.source_transform = source_transform
		self.target_transform = target_transform
		print(len(self.mask_fnames))

	def __len__(self):
		return len(self.mask_fnames)

	def mask_path_to_image_path(self, mask_path):
		filename = mask_path.split('/')[-1]
		file_num, file_ext = filename.split('.')
		return os.path.join(self.db_path, self.image_dir_prefix, str(int(file_num)) + '.jpg')

	def __getitem__(self, idx):
		mask_fname = self.mask_fnames[idx]
		image_fname = self.mask_path_to_image_path(mask_path=mask_fname)
		rgb_image = Image.open(image_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		return input_image, output_image, image_fname


# class RGBSuperMixedResDataset(Dataset):
# 	def __init__(self, db_path, image_dir_prefix, mask_dir_prefix, source_transform, target_transform, resolution=256):
# 		self.db_path = db_path
# 		self.image_dir_prefix = image_dir_prefix
# 		self.mask_dir_prefix = mask_dir_prefix
# 		self.mask_fnames = list(glob.glob(os.path.join(self.db_path, self.mask_dir_prefix, '*/*.png')))
# 		self.source_transform = source_transform
# 		self.target_transform = target_transform
# 		print(len(self.mask_fnames))
#
# 	def __len__(self):
# 		return len(self.mask_fnames)
#
# 	def mask_path_to_image_path(self, mask_path):
# 		filename = mask_path.split('/')[-1]
# 		file_num, file_ext = filename.split('.')
# 		return os.path.join(self.db_path, self.image_dir_prefix, str(int(file_num)) + '.jpg')
#
# 	def __getitem__(self, idx):
# 		mask_fname = self.mask_fnames[idx]
# 		image_fname = self.mask_path_to_image_path(mask_path=mask_fname)
# 		rgb_image = Image.open(image_fname)
# 		input_image = self.source_transform(rgb_image)
# 		output_image = self.target_transform(rgb_image)
# 		return input_image, output_image


class RGBSuperResGeneratedDataset(Dataset):
	def __init__(self, db_path, source_transform, target_transform, resolution=256, filelist=None):
		self.db_path = db_path
		self.img_fnames = list(glob.glob(os.path.join(self.db_path, '*/*.png')))
		self.source_transform = source_transform
		self.target_transform = target_transform
		print(len(self.img_fnames))

	def __len__(self):
		return len(self.img_fnames)

	def img_path_to_latent_path(self, img_path):
		return img_path.replace('/img_', '/latents_').replace('.png', '.npz')

	def __getitem__(self, idx):
		img_fname = self.img_fnames[idx]
		latent_fname = self.img_path_to_latent_path(img_fname)
		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		with np.load(latent_fname) as data:
			true_style_vectors = data['style_vectors']
			true_wplus = data['wplus']
		return input_image, output_image, true_style_vectors, true_wplus, img_fname


class RGBSuperResRealGeneratedDataset(Dataset):
	def __init__(self, real_db_base_path, generated_db_path, source_transform, target_transform, style_vector_ndim, wplus_shape, mask_dir_prefix, image_dir_prefix, resolution=256, filelist=None):
		self.real_db_base_path = real_db_base_path
		self.generated_db_path = generated_db_path
		self.mask_dir_prefix = mask_dir_prefix
		self.image_dir_prefix = image_dir_prefix
		self.real_img_fnames = list(glob.glob(os.path.join(self.real_db_base_path, self.mask_dir_prefix, '*/*.png')))
		self.generated_img_fnames = list(glob.glob(os.path.join(self.generated_db_path, '*/*.png')))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.style_vector_shape = (1, style_vector_ndim)
		self.wplus_shape = (wplus_shape[0], wplus_shape[1])
		print(len(self.real_img_fnames), len(self.generated_img_fnames))

	def __len__(self):
		return len(self.generated_img_fnames)

	def img_path_to_latent_path(self, img_path):
		return img_path.replace('/img_', '/latents_').replace('.png', '.npz')

	def mask_path_to_image_path(self, mask_path):
		filename = mask_path.split('/')[-1]
		file_num, file_ext = filename.split('.')
		return os.path.join(self.real_db_base_path, self.image_dir_prefix, str(int(file_num)) + '.jpg')

	def __getitem__(self, idx):
		true_style_vectors = None
		true_wplus = None
		if random.uniform(0, 1) < 0.7:
			img_fname = self.generated_img_fnames[idx]
			latent_fname = self.img_path_to_latent_path(img_fname)
			with np.load(latent_fname) as data:
				true_style_vectors = torch.Tensor(data['style_vectors'])
				true_wplus = torch.Tensor(data['wplus'])
		else:
			if idx >= len(self.real_img_fnames):
				idx = random.randrange(len(self.real_img_fnames))
			img_fname = self.mask_path_to_image_path(self.real_img_fnames[idx])
			true_style_vectors = torch.ones(self.style_vector_shape).fill_(float('nan'))
			true_wplus = torch.ones(self.wplus_shape).fill_(float('nan'))


		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		return input_image, output_image, true_style_vectors, true_wplus, img_fname


class CLEVRSuperResRealGeneratedDataset(Dataset):
	def __init__(self, real_db_base_path, generated_db_path, source_transform, target_transform, style_vector_ndim, wplus_shape, image_dir_prefix=None, resolution=256, max_size=None):
		self.generated_db_path = generated_db_path
		self.image_dir_prefix = image_dir_prefix
		if real_db_base_path is not None:
			self.real_db_base_path = real_db_base_path
			self.real_img_fnames = sorted(list(glob.glob(os.path.join(self.real_db_base_path,'*.png'))))
			if max_size is not None:
				self.real_img_fnames = self.real_img_fnames[0:max_size]
			print(f'Size of real dataset: {len(self.real_img_fnames)}')
		self.generated_img_fnames = list(glob.glob(os.path.join(self.generated_db_path, '*/*.png')))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.style_vector_shape = (1, style_vector_ndim)
		self.wplus_shape = (wplus_shape[0], wplus_shape[1])
		print(len(self.generated_img_fnames))

	def __len__(self):
		return len(self.generated_img_fnames)

	def img_path_to_latent_path(self, img_path):
		return img_path.replace('/img_', '/latents_').replace('.png', '.npz')

	def __getitem__(self, idx):
		true_style_vectors = None
		true_wplus = None
		img_fname = self.generated_img_fnames[idx]
		latent_fname = self.img_path_to_latent_path(img_fname)
		if random.uniform(0, 1) < 0.7:
			img_fname = self.generated_img_fnames[idx]
			latent_fname = self.img_path_to_latent_path(img_fname)
			with np.load(latent_fname) as data:
				true_style_vectors = torch.Tensor(data['style_vectors'])
				true_wplus = torch.Tensor(data['wplus'])
		else:
			if idx >= len(self.real_img_fnames):
				idx = random.randrange(len(self.real_img_fnames))
			img_fname = self.real_img_fnames[idx]
			true_style_vectors = torch.ones(self.style_vector_shape).fill_(float('nan'))
			true_wplus = torch.ones(self.wplus_shape).fill_(float('nan'))

		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		return input_image, output_image, true_style_vectors, true_wplus, img_fname


class CLEVRSuperResGeneratedDataset(Dataset):
	def __init__(self, generated_db_path, source_transform, target_transform, style_vector_ndim=None, wplus_shape=None, image_dir_prefix=None, resolution=256, max_size=None, shuffle=True):
		self.generated_db_path = generated_db_path
		self.image_dir_prefix = image_dir_prefix
		# df = pd.read_csv(generated_db_path)
		# df = df.sample(frac=1)
		# self.generated_img_fnames = list(df.filename.values)
		if image_dir_prefix is None:
			self.generated_img_fnames = glob.glob(os.path.join(generated_db_path, '*/*.png'))
		else:
			df = pd.read_csv(generated_db_path)
			if shuffle:
				df = df.sample(frac=1)
			self.generated_img_fnames = list(df.filename.values)

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.style_vector_shape = (1, style_vector_ndim)
		self.wplus_shape = (wplus_shape[0], wplus_shape[1])
		print(len(self.generated_img_fnames))

	def __len__(self):
		return len(self.generated_img_fnames)

	def img_path_to_latent_path(self, img_path):
		return img_path.replace('/img_', '/latents_').replace('.png', '.npz')

	def __getitem__(self, idx):
		true_style_vectors = None
		true_wplus = None
		if self.image_dir_prefix is None:
			img_fname = self.generated_img_fnames[idx]
		else:
			img_fname = os.path.join(self.image_dir_prefix, self.generated_img_fnames[idx])
		latent_fname = self.img_path_to_latent_path(img_fname)
		with np.load(latent_fname) as data:
			true_style_vectors = torch.Tensor(data['style_vectors'])
			true_wplus = torch.Tensor(data['wplus'])

		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		return input_image, output_image, true_style_vectors, true_wplus, img_fname

class RGBSuperResBackprojectedDataset(Dataset):
	def __init__(self, generated_db_path, source_transform, target_transform, style_vector_ndim=None, wplus_shape=None, image_dir_prefix=None, resolution=256, max_size=None, shuffle=True):
		self.generated_db_path = generated_db_path
		self.image_dir_prefix = image_dir_prefix
		if image_dir_prefix is None:
			self.generated_img_fnames = glob.glob(os.path.join(generated_db_path, '*/*iteration_300*.png'))
		else:
			df = pd.read_csv(generated_db_path)
			if shuffle:
				df = df.sample(frac=1)
			self.generated_img_fnames = list(df.filename.values)

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.real_data_prefix = '/run/user/61513/loopmnt1/CelebA-HQ-img/'

		print(len(self.generated_img_fnames))

	def __len__(self):
		return len(self.generated_img_fnames)

	def gen_img_to_real_img_path(self, gen_img_path):
		return os.path.join(self.real_data_prefix, gen_img_path.split('/')[-1].split('-')[0]+'.jpg')

	def img_path_to_latent_path(self, img_path):
		return img_path.replace('-project_', '-latents_').replace('.png', '.npz')

	def __getitem__(self, idx):
		true_style_vectors = None
		true_wplus = None
		if self.image_dir_prefix is None:
			img_fname = self.gen_img_to_real_img_path(self.generated_img_fnames[idx])


		else:
			img_fname = os.path.join(self.image_dir_prefix, self.generated_img_fnames[idx])
		latent_fname = self.img_path_to_latent_path(self.generated_img_fnames[idx])
		with np.load(latent_fname) as data:
			true_style_vectors = torch.Tensor(data['optimized_sv'])
			true_wplus = torch.Tensor(data['optimized_w'])


		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		return input_image, output_image, true_style_vectors, true_wplus, img_fname

class CLEVRSuperResBackprojectedDataset(Dataset):
	def __init__(self, generated_db_path, source_transform, target_transform, style_vector_ndim=None, wplus_shape=None, image_dir_prefix=None, resolution=256, max_size=None, shuffle=True):
		self.generated_db_path = generated_db_path
		self.image_dir_prefix = image_dir_prefix
		# df = pd.read_csv(generated_db_path)
		# df = df.sample(frac=1)
		# self.generated_img_fnames = list(df.filename.values)
		if image_dir_prefix is None:
			self.generated_img_fnames = glob.glob(os.path.join(generated_db_path, '*/*iteration_300*.png'))
		else:
			df = pd.read_csv(generated_db_path)
			if shuffle:
				df = df.sample(frac=1)
			self.generated_img_fnames = list(df.filename.values)

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.style_vector_shape = (1, style_vector_ndim)
		self.wplus_shape = (wplus_shape[0], wplus_shape[1])
		self.real_data_prefix = '/home/gridsan/swamiviv/datasets/CLEVR/CLEVR_simple/images'
		print(len(self.generated_img_fnames))

	def __len__(self):
		return len(self.generated_img_fnames)

	def img_path_to_latent_path(self, img_path):
		return img_path.replace('-project_', '-latents_').replace('.png', '.npz')

	def gen_img_path_to_real_path(self, img_path):
		return os.path.join(self.real_data_prefix, img_path.split('/')[-1].split('-')[0] + '.png')

	def __getitem__(self, idx):
		true_style_vectors = None
		true_wplus = None
		if self.image_dir_prefix is None:
			#img_fname = self.gen_img_path_to_real_path(self.generated_img_fnames[idx])
			img_fname = self.generated_img_fnames[idx]

		else:
			img_fname = os.path.join(self.image_dir_prefix, self.generated_img_fnames[idx])
		latent_fname = self.img_path_to_latent_path(self.generated_img_fnames[idx])
		with np.load(latent_fname) as data:
			true_style_vectors = torch.Tensor(data['optimized_sv'])
			true_wplus = torch.Tensor(data['optimized_w'])


		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		return input_image, output_image, true_style_vectors, true_wplus, img_fname

class CLEVRSuperResRealDataset(Dataset):
	def __init__(self, real_db_path, source_transform, target_transform, style_vector_ndim=None, wplus_shape=None, image_dir_prefix=None, resolution=256, max_size=None, shuffle=True):
		self.real_db_path = real_db_path
		self.image_dir_prefix = image_dir_prefix
		if image_dir_prefix is None:
			self.real_img_fnames = glob.glob(os.path.join(real_db_path, '*.png'))
		else:
			df = pd.read_csv(real_db_path)
			if shuffle:
				df = df.sample(frac=1)
			self.generated_img_fnames = list(df.filename.values)

		self.source_transform = source_transform
		self.target_transform = target_transform
		print(len(self.real_img_fnames))

	def __len__(self):
		return len(self.real_img_fnames)

	def __getitem__(self, idx):
		true_style_vectors = None
		true_wplus = None
		if self.image_dir_prefix is None:
			img_fname = self.real_img_fnames[idx]
		else:
			img_fname = os.path.join(self.image_dir_prefix, self.real_img_fnames[idx])

		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		output_image = self.target_transform(rgb_image)
		return input_image, output_image, img_fname


class RGBInpaintGeneratedDataset(Dataset):
	def __init__(self, db_path, source_transform, target_transform, mask_thresholds=None, resolution=256, filelist=None):
		self.db_path = db_path
		self.img_fnames = list(glob.glob(os.path.join(self.db_path, '*/*.png')))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.mask_thresholds = mask_thresholds
		print(len(self.img_fnames))

	def __len__(self):
		return len(self.img_fnames)

	def img_path_to_latent_path(self, img_path):
		return img_path.replace('/img_', '/latents_').replace('.png', '.npz')

	def __getitem__(self, idx):
		img_fname = self.img_fnames[idx]
		latent_fname = self.img_path_to_latent_path(img_fname)
		rgb_image = Image.open(img_fname)
		input_image = self.source_transform(rgb_image)
		input_image, input_mask = mask_upsample(
			input_image.unsqueeze(0),
			mask_cent=0.5,
			threshold=np.random.uniform(low=self.mask_thresholds['low'], high=self.mask_thresholds['high']),
		)
		input_image = torch.cat((input_image.squeeze(0), input_mask.squeeze(0)), axis=0)
		output_image = self.target_transform(rgb_image)
		with np.load(latent_fname) as data:
			true_style_vectors = data['style_vectors']
			true_wplus = data['wplus']
		return input_image, output_image, true_style_vectors, true_wplus, img_fname
