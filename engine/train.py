from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device, scaler: torch.cuda.amp.GradScaler):
	model.train()
	running_loss = 0.0
	running_correct = 0
	total = 0
	criterion = torch.nn.CrossEntropyLoss()
	for images, targets in tqdm(loader, desc="train", leave=False):
		images = images.to(device)
		targets = targets.to(device)
		optimizer.zero_grad(set_to_none=True)
		with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
			outputs = model(images)
			loss = criterion(outputs, targets)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		running_loss += loss.item() * images.size(0)
		_, preds = outputs.max(1)
		running_correct += preds.eq(targets).sum().item()
		total += images.size(0)

	return {
		"loss": running_loss / total,
		"acc": running_correct / total,
	}


def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
	model.eval()
	criterion = torch.nn.CrossEntropyLoss()
	running_loss = 0.0
	running_correct = 0
	total = 0
	with torch.no_grad():
		for images, targets in tqdm(loader, desc="eval", leave=False):
			images = images.to(device)
			targets = targets.to(device)
			outputs = model(images)
			loss = criterion(outputs, targets)
			running_loss += loss.item() * images.size(0)
			_, preds = outputs.max(1)
			running_correct += preds.eq(targets).sum().item()
			total += images.size(0)
	return {
		"loss": running_loss / total,
		"acc": running_correct / total,
	}


def save_checkpoint(path: str, model, optimizer, epoch: int, best_acc: float):
	torch.save({
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"epoch": epoch,
		"best_acc": best_acc,
	}, path)
