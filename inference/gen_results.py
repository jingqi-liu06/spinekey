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
    运行完整的脊椎检测流程
    
    Args:
        images_dir: 输入图片目录
        cfg_det_path: 检测模型配置文件路径
        ckpt_det_path: 检测模型权重文件路径
        anns_path: 原始标注文件路径
        output_dir: 输出目录
        cuda_id: GPU设备ID
        iou_threshold: IOU阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径
    results_path = os.path.join(output_dir, 'results.json')
    gts_path = os.path.join(output_dir, 'gts.json')
    iou_results_path = os.path.join(output_dir, 'iou_results.json')
    
    # 1. 生成检测结果
    print("正在生成检测结果...")
    results = detect_vertebrae_det(
        images_dir=images_dir,
        cfg_det_path=cfg_det_path,
        ckpt_det_path=ckpt_det_path,
        results_path=results_path,
        cuda_id=cuda_id
    )
    
    # 2. 生成真实标注（ground truth）
    print("正在生成真实标注...")
    create_gts(anns_path, gts_path)
    
    # 3. 计算IOU并生成最终结果
    print("正在计算IOU...")
    process_annotations(
        gts_path=gts_path,
        results_path=results_path,
        output_path=iou_results_path,
        iou_threshold=iou_threshold
    )
    
    print(f"处理完成！结果已保存到: {output_dir}")
    return {
        'results_path': results_path,
        'gts_path': gts_path,
        'iou_results_path': iou_results_path
    }

if __name__ == "__main__":
    # 示例用法
    config = {
        'images_dir': "/hdd/srt19/data/xray/compression_fracture/压缩性骨折图片/疑似骨折全脊柱_images",  # 替换为您的图片目录
        'cfg_det_path': "/home/srt19/jingqi/ver_det/wkdir-quanjizhui_cascade/det_cascade-mask-rcnn.py",  # 替换为您的配置文件路径
        'ckpt_det_path': "/home/srt19/jingqi/ver_det/wkdir-quanjizhui_cascade/best_coco_segm_mAP_epoch_5.pth",  # 替换为您的检查点路径
        'anns_path': "/hdd/srt19/data/xray/compression_fracture/压缩性骨折json/疑似骨折全脊柱.json",  # 替换为您的标注文件路径
        'output_dir': "/hdd/srt19/data/疑似骨折全脊柱/",  # 替换为您想要保存结果的目录
        'cuda_id': 5,
        'iou_threshold': 0.5
    }
    
    results = run_vertebrae_detection_pipeline(**config)