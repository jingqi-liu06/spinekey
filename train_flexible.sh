#!/usr/bin/env bash
# Flexible training script with GPU selection support
# Usage examples:
#   ./train_flexible.sh                           # Use default GPUs (0,1,2,3,4,5)
#   ./train_flexible.sh --gpus 0,1,2,3            # Use specific GPUs
#   ./train_flexible.sh --gpus 1,3,5 --port 29501 # Use specific GPUs and port

# Default configuration
DEFAULT_GPUS="0,1,2,3,4,5"
DEFAULT_PORT="29500"
DEFAULT_WORK_DIR="work-dir/"

# Initialize variables
GPUS=$DEFAULT_GPUS
PORT=$DEFAULT_PORT
WORK_DIR=$DEFAULT_WORK_DIR

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus        Comma-separated list of GPU IDs (default: $DEFAULT_GPUS)"
            echo "  --port        Master port for distributed training (default: $DEFAULT_PORT)"
            echo "  --work-dir    Working directory for outputs (default: $DEFAULT_WORK_DIR)"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --gpus 0,1,2,3"
            echo "  $0 --gpus 1,3,5 --port 29501"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Count number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Validate GPU count
if [[ $NUM_GPUS -eq 0 ]]; then
    echo "Error: No GPUs specified"
    exit 1
fi

# Print configuration
echo "==================================="
echo "Training Configuration"
echo "==================================="
echo "GPUs: $GPUS (Count: $NUM_GPUS)"
echo "Master Port: $PORT"
echo "Work Directory: $WORK_DIR"
echo "==================================="

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=$GPUS

# Run training
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train/train_mmdet.py \
    train/det_cascade-mask-rcnn.py \
    --work-dir $WORK_DIR \
    --launcher pytorch

echo "Training completed!"
