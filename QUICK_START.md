# SpineKey Quick Start Guide

## 🚀 Quick Start

### 1. Prepare Data

Organize your data according to the following structure:

```text
data/
├── images/              # All image files
├── det_train.json       # Training annotation file
└── det_test.json        # Testing annotation file
```

### 2. Download Model Weights

```bash
# Create weights directory
mkdir -p checkpoints

# Download pre-trained weights (ResNeXt-101 backbone)
wget https://download.openmmlab.com/mmdetection/v2.0/resnext/resnext101_64x4d.pth -P checkpoints/

# If you have a complete trained model, please name it full_inference.pth and place it in the checkpoints/ directory
```

## 🔧 Train Model

### Step 1: Modify Data Path

Edit line 237 in the `train/det_cascade-mask-rcnn.py` file:

```python
# Change this line:
data_root = "../data/full/"

# Modify to your actual data path:
data_root = "path/to/your/data/"  # For example: "../data/" or "/absolute/path/to/data/"
```

### Step 2: Start Training

```bash
# Basic training (using default GPUs: 0,1,2,3,4,5)
bash train.sh

# Or customize GPU configuration
bash train_flexible.sh --gpus 0,1,2,3 --port 29501
```

Training results will be saved in the `work_dir/` directory.

## 🔍 Model Inference

### Step 1: Check Inference Script Configuration

Edit the `inference.sh` file and confirm the following paths are correct:

```bash
# Main configuration items (modify according to actual situation)
IMAGES_DIR="../data/images"                    # Test image directory
CFG_DET_PATH="inference/inference.py"          # Inference configuration file
CKPT_DET_PATH="checkpoints/full_inference.pth" # Model weight file  
ANNS_PATH="../data/det_test.json"              # Test annotation file
OUTPUT_DIR="results/"                          # Output directory
```

### Step 2: Execute Inference

```bash
# Use default configuration
bash inference.sh

# Or customize parameters
bash inference.sh \
    --images_dir data/images \
    --cfg_det_path inference/inference.py \
    --ckpt_det_path checkpoints/full_inference.pth \
    --anns_path data/det_test.json \
    --output_dir results/ \
    --cuda_id 0
```

## 📊 Generate Evaluation Metrics

### Step 1: Modify Evaluation Script Paths

Edit the `inference/generate_indicators.sh` file:

```bash
python evaluate_keypoints.py \
    --gt_file ../data/gts.json \                    # Ground truth annotation file
    --result_file ../data/iou_results.json \        # Inference result file
    --output ../data/evaluation_indicators.txt      # Evaluation report output file
```

Modify the paths to your actual paths, for example:

```bash
python evaluate_keypoints.py \
    --gt_file results/gts.json \
    --result_file results/iou_results.json \
    --output results/evaluation_indicators.txt
```

### Step 2: Generate Evaluation Report

```bash
# Enter inference directory
cd inference/

# Execute evaluation
bash generate_indicators.sh
```

Evaluation results will include the following metrics:

- **Detection Accuracy**: AP, AP50, AP75
- **Keypoint Accuracy**: PCK, AUC, EPE

## ⚠️ Common Issues

### Issue 1: Path Errors

```bash
# Check if data exists
ls -la data/images/
ls -la data/det_train.json
ls -la data/det_test.json

# Check weight files
ls -la checkpoints/
```

### Issue 2: GPU Memory Insufficient

Edit `train/det_cascade-mask-rcnn.py`:

```python
# Reduce batch size
train_dataloader = dict(batch_size=4)  # Change from 8 to 4
val_dataloader = dict(batch_size=2)    # Change from 4 to 2
```

### Issue 3: GPU Device Issues

```bash
# Check available GPUs
nvidia-smi

# Modify GPUs to use
export CUDA_VISIBLE_DEVICES=0,1
```

## 📁 Important File Descriptions

| File/Directory | Description |
|-----------|------|
| `train.sh` | Training launch script |
| `train/det_cascade-mask-rcnn.py` | Training configuration file (need to modify data_root) |
| `inference.sh` | Inference launch script (need to modify path configurations) |
| `inference/inference.py` | Inference configuration file |
| `inference/generate_indicators.sh` | Evaluation script (need to modify paths) |
| `checkpoints/` | Model weights directory |
| `data/` | Data directory |
| `work_dir/` | Training output directory |

## 🎯 Complete Workflow Example

The following is a complete usage workflow example:

```bash
# 1. Prepare data (assuming data is in ./data/ directory)
ls data/
# Output should show: images/  det_train.json  det_test.json

# 2. Download weights
mkdir -p checkpoints
wget https://download.openmmlab.com/mmdetection/v2.0/resnext/resnext101_64x4d.pth -P checkpoints/

# 3. Modify data_root in training configuration to "../data/"

# 4. Start training
bash train.sh

# 5. After training is complete, modify path configurations in inference.sh

# 6. Execute inference
bash inference.sh \
    --images_dir data/images \
    --anns_path data/det_test.json \
    --output_dir results/

# 7. Modify paths in inference/generate_indicators.sh

# 8. Generate evaluation metrics
cd inference/
bash generate_indicators.sh
```

## 📞 Need Help?

If you encounter problems, please check:

1. Whether all file paths are correct
2. Whether data format complies with COCO standards  
3. Whether GPU memory is sufficient
4. Whether dependency packages are installed completely

It is recommended to backup original files before modifying configuration files.
