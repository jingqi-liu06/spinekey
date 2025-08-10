#!/bin/bash

# Vertebrae Detection Pipeline Inference Script
# This script runs the complete vertebrae detection pipeline

# Default Configuration
IMAGES_DIR="data/full/origin"
CFG_DET_PATH="inference/inference.py"
CKPT_DET_PATH="checkpoints/full_inference.pth"
ANNS_PATH="data/full/annotations.json" # This is the annotation json file for the data to be detected
OUTPUT_DIR="data/full/output/"
CUDA_ID=5 # using GPU ID 
IOU_THRESHOLD=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --images_dir)
            IMAGES_DIR="$2"
            shift 2
            ;;
        --cfg_det_path)
            CFG_DET_PATH="$2"
            shift 2
            ;;
        --ckpt_det_path)
            CKPT_DET_PATH="$2"
            shift 2
            ;;
        --anns_path)
            ANNS_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cuda_id)
            CUDA_ID="$2"
            shift 2
            ;;
        --iou_threshold)
            IOU_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --images_dir      Input images directory"
            echo "  --cfg_det_path    Detection model configuration file path"
            echo "  --ckpt_det_path   Detection model checkpoint file path"
            echo "  --anns_path       Original annotation file path"
            echo "  --output_dir      Output directory"
            echo "  --cuda_id         GPU device ID (default: 5)"
            echo "  --iou_threshold   IOU threshold (default: 0.5)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --images_dir /path/to/images --output_dir /path/to/output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required paths
if [[ ! -d "$IMAGES_DIR" ]]; then
    echo "Error: Images directory does not exist: $IMAGES_DIR"
    exit 1
fi

if [[ ! -f "$CFG_DET_PATH" ]]; then
    echo "Error: Detection config file does not exist: $CFG_DET_PATH"
    exit 1
fi

if [[ ! -f "$CKPT_DET_PATH" ]]; then
    echo "Error: Detection checkpoint file does not exist: $CKPT_DET_PATH"
    exit 1
fi

if [[ ! -f "$ANNS_PATH" ]]; then
    echo "Error: Annotation file does not exist: $ANNS_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "==================================="
echo "Vertebrae Detection Pipeline"
echo "==================================="
echo "Images Directory: $IMAGES_DIR"
echo "Config File: $CFG_DET_PATH"
echo "Checkpoint File: $CKPT_DET_PATH"
echo "Annotations File: $ANNS_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "CUDA Device ID: $CUDA_ID"
echo "IOU Threshold: $IOU_THRESHOLD"
echo "==================================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Run the Python script with arguments
python3 << EOF
import sys
import os
sys.path.append('$SCRIPT_DIR')
from gen_results import run_vertebrae_detection_pipeline

config = {
    'images_dir': '$IMAGES_DIR',
    'cfg_det_path': '$CFG_DET_PATH',
    'ckpt_det_path': '$CKPT_DET_PATH',
    'anns_path': '$ANNS_PATH',
    'output_dir': '$OUTPUT_DIR',
    'cuda_id': $CUDA_ID,
    'iou_threshold': $IOU_THRESHOLD
}

try:
    results = run_vertebrae_detection_pipeline(**config)
    print("\n==================================="
    print("Pipeline completed successfully!")
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("===================================")
except Exception as e:
    print(f"Error running pipeline: {e}")
    sys.exit(1)
EOF

# Check if Python script succeeded
if [[ $? -eq 0 ]]; then
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Inference failed!"
    exit 1
fi
