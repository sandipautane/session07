import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.stats import compute_cifar10_mean_std, save_stats, load_stats
from utils.transforms import build_albu_transforms
from data.albu_dataset import CIFAR10Albumentations
from models.dws_dilated_cnn import SmallDWSDilatedCNN
from engine.train import train_one_epoch, evaluate, save_checkpoint
from utils.plots import plot_curves
from torchinfo import summary


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_root", type=str, default="./data")
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--stats_path", type=str, default="./artifacts/cifar10_stats.json")
	parser.add_argument("--checkpoint", type=str, default="./artifacts/best.pt")
	args = parser.parse_args()

	# Ensure SSL verification uses certifi CA bundle (helps on macOS)
	try:
		import os, certifi
		ca = certifi.where()
		os.environ.setdefault("SSL_CERT_FILE", ca)
		os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
	except Exception:
		pass

	artifacts_dir = str(Path(args.checkpoint).parent)
	Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
	Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)

	if not Path(args.stats_path).exists():
		mean, std = compute_cifar10_mean_std(args.data_root, batch_size=512, num_workers=args.workers)
		save_stats(args.stats_path, mean, std)
	else:
		mean, std = load_stats(args.stats_path)

	mean_01 = tuple(float(m.item()) for m in mean)
	std_01 = tuple(float(s.item()) for s in std)

	train_tfms, test_tfms = build_albu_transforms(mean_01, std_01)

	train_ds = CIFAR10Albumentations(args.data_root, train=True, transform=train_tfms, download=True)
	test_ds = CIFAR10Albumentations(args.data_root, train=False, transform=test_tfms, download=True)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SmallDWSDilatedCNN().to(device)
	# Print model summary
	print(summary(model, input_size=(1, 3, 32, 32), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), depth=3))

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

	best_acc = 0.0
	train_losses, val_losses = [], []
	train_accs, val_accs = [], []
	lrs = []
	for epoch in range(1, args.epochs + 1):
		# current LR from optimizer
		current_lr = optimizer.param_groups[0]["lr"]
		lrs.append(current_lr)

		train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
		eval_metrics = evaluate(model, test_loader, device)

		train_losses.append(train_metrics["loss"]) 
		val_losses.append(eval_metrics["loss"]) 
		train_accs.append(train_metrics["acc"]) 
		val_accs.append(eval_metrics["acc"]) 

		print(
			f"Epoch {epoch}: lr={current_lr:.6f} | "
			f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
			f"val loss={eval_metrics['loss']:.4f} acc={eval_metrics['acc']:.4f}"
		)
		if eval_metrics["acc"] > best_acc:
			best_acc = eval_metrics["acc"]
			save_checkpoint(args.checkpoint, model, optimizer, epoch, best_acc)

	# save plots
	plot_curves(train_losses, val_losses, train_accs, val_accs, artifacts_dir)
	print(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
	main()
