from typing import List
from pathlib import Path
import matplotlib.pyplot as plt


def plot_curves(train_losses: List[float], val_losses: List[float], train_accs: List[float], val_accs: List[float], out_dir: str):
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	# Loss plot
	plt.figure()
	plt.plot(train_losses, label="train_loss")
	plt.plot(val_losses, label="val_loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(str(Path(out_dir) / "loss.png"))
	plt.close()
	# Accuracy plot
	plt.figure()
	plt.plot(train_accs, label="train_acc")
	plt.plot(val_accs, label="val_acc")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(str(Path(out_dir) / "accuracy.png"))
	plt.close()
