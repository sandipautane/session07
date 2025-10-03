import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def compute_cifar10_mean_std(data_root: str, batch_size: int = 512, num_workers: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute mean and std over the CIFAR-10 training set in [0, 1] range.
	Returns tensors of shape (3,) for mean and std.
	"""
	# Use ToTensor to convert PIL to [0,1] tensor for precise stats
	dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.ToTensor())
	loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

	n_pixels = 0
	channel_sum = torch.zeros(3, dtype=torch.float64)
	channel_sq_sum = torch.zeros(3, dtype=torch.float64)

	for imgs, _ in loader:
		# imgs: [B, 3, H, W] in [0,1]
		b, c, h, w = imgs.shape
		n_pixels += b * h * w
		channel_sum += imgs.sum(dim=[0, 2, 3]).to(dtype=torch.float64)
		channel_sq_sum += (imgs ** 2).sum(dim=[0, 2, 3]).to(dtype=torch.float64)

	mean = (channel_sum / n_pixels).to(dtype=torch.float32)
	std = torch.sqrt((channel_sq_sum / n_pixels) - (mean.double() ** 2)).to(dtype=torch.float32)
	return mean, std


def save_stats(stats_path: str, mean: torch.Tensor, std: torch.Tensor) -> None:
	Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
	with open(stats_path, "w") as f:
		json.dump({
			"mean": mean.tolist(),
			"std": std.tolist()
		}, f)


def load_stats(stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
	with open(stats_path, "r") as f:
		data = json.load(f)
	mean = torch.tensor(data["mean"], dtype=torch.float32)
	std = torch.tensor(data["std"], dtype=torch.float32)
	return mean, std
