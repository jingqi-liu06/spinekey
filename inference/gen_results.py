import os
from utils.generate_results_det import detect_vertebrae_det
from utils.create_gts import create_gts
from utils.cal_iou import process_annotations

def run_vertebrae_detection_pipeline(
    images_dir,
    cfg_det_path,
    ckpt_det_path,
    anns_path,
    output_dir,
    cuda_id=0,
    iou_threshold=0.5
):
    """
    Run complete vertebrae detection pipeline
    
    Args:
        images_dir: Input images directory
        cfg_det_path: Detection model configuration file path
        ckpt_det_path: Detection model checkpoint file path
        anns_path: Original annotation file path
        output_dir: Output directory
        cuda_id: GPU device ID
        iou_threshold: IOU threshold
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file paths
    results_path = os.path.join(output_dir, 'results.json')
    gts_path = os.path.join(output_dir, 'gts.json')
    iou_results_path = os.path.join(output_dir, 'iou_results.json')
    
    # 1. Generate detection results
    print("Generating detection results...")
    results = detect_vertebrae_det(
        images_dir=images_dir,
        cfg_det_path=cfg_det_path,
        ckpt_det_path=ckpt_det_path,
        results_path=results_path,
        cuda_id=cuda_id
    )
    
    # 2. Generate ground truth annotations
    print("Generating ground truth annotations...")
    create_gts(anns_path, gts_path)
    
    # 3. Calculate IOU and generate final results
    print("Calculating IOU...")
    process_annotations(
        gts_path=gts_path,
        results_path=results_path,
        output_path=iou_results_path,
        iou_threshold=iou_threshold
    )
    
    print(f"Processing completed! Results saved to: {output_dir}")
    return {
        'results_path': results_path,
        'gts_path': gts_path,
        'iou_results_path': iou_results_path
    }

if __name__ == "__main__":
    # Example usage
    config = {
        'images_dir': "/hdd/srt19/data/xray/compression_fracture/压缩性骨折图片/疑似骨折全脊柱_images",  # Replace with your images directory
        'cfg_det_path': "/home/srt19/jingqi/ver_det/wkdir-quanjizhui_cascade/det_cascade-mask-rcnn.py",  # Replace with your config file path
        'ckpt_det_path': "/home/srt19/jingqi/ver_det/wkdir-quanjizhui_cascade/best_coco_segm_mAP_epoch_5.pth",  # Replace with your checkpoint path
        'anns_path': "/hdd/srt19/data/xray/compression_fracture/压缩性骨折json/疑似骨折全脊柱.json",  # Replace with your annotation file path
        'output_dir': "/hdd/srt19/data/疑似骨折全脊柱/",  # Replace with your desired output directory
        'cuda_id': 5,
        'iou_threshold': 0.5
    }
    
    results = run_vertebrae_detection_pipeline(**config)