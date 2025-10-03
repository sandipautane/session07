from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_albu_transforms(mean_01: Tuple[float, float, float], std_01: Tuple[float, float, float],
							coarse_size: int = 16,
							p_flip: float = 0.5,
							p_ssr: float = 0.8,
							p_coarse: float = 0.5):
	"""
	Build Albumentations pipeline with:
	- HorizontalFlip
	- ShiftScaleRotate
	- CoarseDropout with 1 hole of 16x16, fill_value = dataset mean
	Normalization is applied after augmentations. Output is CHW tensor.
	"""
	mean_uint8 = tuple(int(round(m * 255.0)) for m in mean_01)

	train_tfms = A.Compose([
		A.HorizontalFlip(p=p_flip),
		A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=0, p=p_ssr),
		A.CoarseDropout(
			max_holes=1,
			max_height=coarse_size,
			max_width=coarse_size,
			min_holes=1,
			min_height=coarse_size,
			min_width=coarse_size,
			fill_value=mean_uint8,
			mask_fill_value=None,
			p=p_coarse,
		),
		A.Normalize(mean=mean_01, std=std_01),
		ToTensorV2(),
	])

	test_tfms = A.Compose([
		A.Normalize(mean=mean_01, std=std_01),
		ToTensorV2(),
	])

	return train_tfms, test_tfms
