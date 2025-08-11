# SpineKey: Spine Vertebra Detection and Keypoint Analysis System

## <span style="color: red">This Repository is still under construction and undergoing ongoing maintenance</span>
## Project Overview

SpineKey is a deep learning-based spine vertebra detection and keypoint analysis system that uses Cascade R-CNN architecture for vertebra detection and provides a complete training, inference, and evaluation pipeline.

## Environment Requirements

The environment for this project is based on **[MMDetection](https://github.com/open-mmlab/mmdetection)**. Please refer to the official documentation for installation instructions.

## Directory Structure

```text
SpineKey/
├── data/                    # Data directory
│   ├── full/               # Complete dataset
│   │   ├── images/         # Image files
│   │   │   └── images/     # Image files
│   │   ├── det_train.json  # Training annotation file
│   │   └── det_test.json   # Testing annotation file
│   └── partial/            # Partial dataset
├── checkpoints/            # Model weights directory
├── train/                  # Training related code
│   ├── train_mmdet.py      # Training main program
│   └── det_cascade-mask-rcnn.py  # Model configuration file
├── inference/              # Inference related code
│   ├── inference.py        # Inference configuration file
│   ├── gen_results.py      # Result generation script
│   ├── evaluate_keypoints.py  # Evaluation script
│   ├── generate_indicators.sh  # Evaluation metrics generation script
│   └── utils/              # Utility functions
├── work_dir/               # Training output directory
├── train.sh                # Training launch script
├── train_flexible.sh       # Flexible training script
└── inference.sh            # Inference launch script
```

## Data Preparation

### 1. Dataset Organization Structure

Ensure your data is organized in the following format:

```text
data/
├── images/              # Image files directory
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── det_train.json       # Training set annotation file (COCO format)
└── det_test.json        # Test set annotation file (COCO format)
```

### 2. Annotation File Format

Annotation files use COCO format and contain the following information:

- Vertebra bounding box (bbox)
- Vertebra mask (mask)
- Keypoint information (keypoints)

## Model Weight Download

Before starting training or inference, please download the pre-trained model weights:

### 1. Pre-trained Backbone Network Weights

Please go to [ModelScope](https://modelscope.cn/models/kyan007/spinekey)
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download ResNeXt-101 pre-trained weights
modelscope download --model 'kyan007/spinekey' --local_dir 'path/to/dir'
```

### 2. Complete Model Weights

If you have trained complete model weights, please place them in the `checkpoints/` directory:

```bash
# Example: Place complete model weights in the specified location
cp your_model_weights.pth checkpoints/full_inference.pth
```

## Training Instructions

### 1. Configure Data Paths

Before starting training, you need to modify the data paths in the configuration file:

Edit the `train/det_cascade-mask-rcnn.py` file and modify the `data_root` parameter on line 237:

```python
# Modify to your actual data path
data_root = "path/to/your/data/"  # For example: "../data/full/" or "/absolute/path/to/data/"
```

### 2. Basic Training

Train using default configuration:

```bash
# Use default GPU configuration (GPU 0,1,2,3,4,5)
bash train.sh
```

### 3. Flexible Training Configuration

Use the `train_flexible.sh` script to customize GPU configuration:

```bash
# View help information
bash train_flexible.sh --help

# Use specified GPUs
bash train_flexible.sh --gpus 0,1,2,3

# Custom port and working directory
bash train_flexible.sh --gpus 1,3,5 --port 29501 --work-dir custom_work_dir/
```

### 4. Training Parameter Description

Main training parameters are located in `train/det_cascade-mask-rcnn.py`:

- `data_root`: Dataset root directory
- `max_epochs`: Maximum training epochs (default 24)
- `batch_size`: Batch size (default training 8, validation 4)
- `lr`: Learning rate (default 0.02)
- `num_classes`: Number of classes (vertebra detection is 1)

### 5. Monitor Training Process

Training logs and model weights will be saved in the `work_dir/` directory:

```bash
# View training logs
tail -f work_dir/$(date +%Y%m%d_%H%M%S).log

# Use tensorboard monitoring (if configured)
tensorboard --logdir work_dir/
```

## Inference Instructions

### 1. Configure Inference Paths

Before performing inference, you need to confirm and modify the path configurations in the `inference.sh` script:

```bash
# Edit inference.sh, modify the following default paths
IMAGES_DIR="data/full/origin"              # Test image directory
CFG_DET_PATH="inference/inference.py"      # Inference configuration file
CKPT_DET_PATH="checkpoints/full_inference.pth"  # Model weight file
ANNS_PATH="data/full/annotations.json"     # Annotation file path
OUTPUT_DIR="data/full/output/"             # Output directory
```

### 2. Execute Inference

```bash
# View inference script help
bash inference.sh --help

# Inference with default configuration
bash inference.sh

# Custom parameter inference
bash inference.sh \
    --images_dir data/full/images \
    --cfg_det_path inference/inference.py \
    --ckpt_det_path checkpoints/full_inference.pth \
    --anns_path data/full/det_test.json \
    --output_dir results/ \
    --cuda_id 0 \
    --iou_threshold 0.5
```

### 3. Inference Output

After inference is completed, the following files will be generated in the output directory:

```text
output_dir/
├── results.json          # Raw detection results
├── gts.json             # Processed ground truth annotations
└── iou_results.json     # IOU calculation results
```

### 4. Generate Evaluation Metrics

Use the `generate_indicators.sh` script to generate detailed evaluation metrics:

```bash
# Modify file paths in generate_indicators.sh
# Edit inference/generate_indicators.sh
python evaluate_keypoints.py \
    --gt_file path/to/your/gts.json \
    --result_file path/to/your/final_results.json \
    --output path/to/your/evaluation_indicators.txt

# Execute evaluation
bash inference/generate_indicators.sh
```

### 5. Evaluation Metrics Description

The evaluation script will calculate the following metrics:

- **Detection Accuracy Metrics**:
  - AP (Average Precision)
  - AP50 (AP at IoU=0.5)
  - AP75 (AP at IoU=0.75)

- **Keypoint Accuracy Metrics**:
  - PCK (Percentage of Correct Keypoints)
  - AUC (Area Under Curve)
  - EPE (End Point Error)

## Common Issues and Solutions

### 1. Out of Memory

If you encounter out of memory during training:

```bash
# Reduce batch size
# Modify in det_cascade-mask-rcnn.py:
train_dataloader = dict(batch_size=4)  # Change from 8 to 4
val_dataloader = dict(batch_size=2)    # Change from 4 to 2

# Or use fewer GPUs
bash train_flexible.sh --gpus 0,1
```

### 2. Path Errors

Ensure all paths use absolute paths or correct relative paths:

```bash
# Check if data paths exist
ls -la data/full/images/
ls -la data/full/det_train.json
ls -la data/full/det_test.json

# Check if model weights exist
ls -la checkpoints/
```

### 3. CUDA Device Issues

```bash
# Check available GPUs
nvidia-smi

# Modify GPU configuration
export CUDA_VISIBLE_DEVICES=0,1  # Specify GPUs to use
```

## Performance Optimization Recommendations

### 1. Training Optimization

- Use mixed precision training to reduce memory usage
- Appropriately adjust learning rate and batch size
- Use data parallelism to accelerate training

### 2. Inference Optimization

- Use TensorRT or ONNX to optimize inference speed
- Process multiple images in batches
- Use appropriate input resolution

## Citation

If this project helps your research, please consider citing:

```bibtex
@misc{spinekey2024,
  title={SpineKey: Vertebrae Detection and Keypoint Analysis System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/spinekey}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

For questions or suggestions, please contact us through:

- If you have any questions about the cal-dis_R/cal-dis.R file, please contact <wdingben@bjmu.edu.cn>
- For other questions, please contact <jingqi.liu03@outlook.com>
- GitHub Issues: [Project Issues Page](https://github.com/yourusername/spinekey/issues)

## References

```bibtex
@misc{spinekey2024,
  title={SpineKey: Vertebrae Detection and Keypoint Analysis System},
  author={Jingqi Liu, Dingben Wang and et al.},
  year={2025},
  url={https://github.com/yourusername/spinekey},
  note={Available at: \url{https://github.com/yourusername/spinekey}}
}
```

## Changelog

### v1.0.0 (2024-08-04)

- Initial release
- Support for vertebra detection and keypoint analysis
- Complete training and inference pipeline
- Detailed evaluation metric calculation
