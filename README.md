## CIFAR-10 Depthwise-Separable Dilated CNN (PyTorch)

This project trains a compact CNN (<200k params) on CIFAR-10 using:
- Depthwise-Separable convolutions
- Dilated convolutions to enlarge receptive field (>44)
- No MaxPool (downsampling via stride-2 convs)
- Albumentations for data augmentation

### Requirements
Install dependencies:
```bash
python -m pip install -r requirements.txt
```

### Data Augmentation (Albumentations)
- HorizontalFlip (p=0.5)
- ShiftScaleRotate (moderate limits)
- CoarseDropout with:
  - max_holes=1, min_holes=1
  - max_height=16, max_width=16
  - min_height=16, min_width=16
  - fill_value=(dataset mean), mask_fill_value=None
- Normalize with CIFAR-10 mean/std (computed automatically on first run)

### Model
- File: `models/dws_dilated_cnn.py`
- Depthwise-Separable blocks + dilations
- Stride-2 convs for downsampling (no MaxPool)
- Receptive field > 44
- ~<200k parameters

### Train
```bash
python train.py \
	--data_root ./data \
	--epochs 30 \
	--batch_size 128 \
	--lr 0.001
```

The script will:
- Compute CIFAR-10 mean/std if missing and cache at `./artifacts/cifar10_stats.json`
- Print model summary via torchinfo
- Print per-epoch: learning rate, training loss/acc, and validation loss/acc
- Save best checkpoint to `./artifacts/best.pt`
- Save plots to `./artifacts/loss.png` and `./artifacts/accuracy.png`

### Files
- `utils/stats.py`: compute/save/load dataset stats
- `utils/transforms.py`: Albumentations pipelines
- `data/albu_dataset.py`: Albumentations dataset wrapper
- `models/dws_dilated_cnn.py`: model definition
- `engine/train.py`: training/evaluation loops and checkpointing
- `utils/plots.py`: plotting utilities
- `train.py`: entrypoint

### Notes
- AMP is enabled automatically on CUDA.
- If you add a scheduler, the printed lr reflects the optimizer’s current lr each epoch.
