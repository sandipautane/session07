from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
		super().__init__()
		padding = ((kernel_size - 1) // 2) * dilation
		self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
									padding=padding, dilation=dilation, groups=in_channels, bias=False)
		self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.depthwise(x)
		x = self.pointwise(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)


class DilatedBlock(nn.Module):
	def __init__(self, channels: int, dilation: int):
		super().__init__()
		self.conv = DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, dilation=dilation)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.conv(x)


class SmallDWSDilatedCNN(nn.Module):
	"""
	CIFAR-10 model with:
	- Only depthwise-separable and dilated convs
	- No maxpool (downsample via stride 2)
	- Receptive field > 44
	- Parameters < 200k
	"""
	def __init__(self, num_classes: int = 10):
		super().__init__()
		# Stem
		self.stem = DepthwiseSeparableConv(3, 32, kernel_size=3, stride=1, dilation=1)  # 32x32
		# Stage 1
		self.s1_1 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, dilation=1)  # 16x16
		self.s1_2 = DilatedBlock(64, dilation=2)  # increase RF
		# Stage 2
		self.s2_1 = DepthwiseSeparableConv(64, 96, kernel_size=3, stride=2, dilation=1)  # 8x8
		self.s2_2 = DilatedBlock(96, dilation=2)
		self.s2_3 = DilatedBlock(96, dilation=4)
		# Stage 3
		self.s3_1 = DepthwiseSeparableConv(96, 128, kernel_size=3, stride=2, dilation=1)  # 4x4
		self.s3_2 = DilatedBlock(128, dilation=2)
		self.s3_3 = DilatedBlock(128, dilation=4)
		# Head
		self.classifier = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(128, num_classes)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		x = self.s1_1(x)
		x = self.s1_2(x)
		x = self.s2_1(x)
		x = self.s2_2(x)
		x = self.s2_3(x)
		x = self.s3_1(x)
		x = self.s3_2(x)
		x = self.s3_3(x)
		return self.classifier(x)


if __name__ == "__main__":
	net = SmallDWSDilatedCNN()
	x = torch.randn(2, 3, 32, 32)
	y = net(x)
	print("Output:", y.shape)
	# Parameter count
	total_params = sum(p.numel() for p in net.parameters())
	print("Params:", total_params)
