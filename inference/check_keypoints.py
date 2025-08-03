import json
import numpy as np
import argparse

def check_keypoints(gt_file, pred_file):
    # 读取文件
    with open(gt_file, 'r') as f:
        ground_truths = json.load(f)
    
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    print("\n检查关键点数量...")
    has_error = False
    
    # 遍历所有图片
    for image_id, gt_keypoints in ground_truths.items():
        if image_id in predictions:
            pred_keypoints = predictions[image_id]
            for key in gt_keypoints:
                if key in pred_keypoints:
                    gt = np.array(gt_keypoints[key])
                    pred = np.array(pred_keypoints[key])
                    
                    # 检查形状
                    if gt.shape != (4, 2) or pred.shape != (4, 2):
                        has_error = True
                        print(f"\n问题图片ID: {image_id}")
                        print(f"关键点类型: {key}")
                        print(f"Ground Truth形状: {gt.shape}")
                        print(f"Prediction形状: {pred.shape}")
                        print(f"Ground Truth内容: {gt}")
                        print(f"Prediction内容: {pred}")
                        print("-" * 50)
    
    if not has_error:
        print("所有图片的关键点数量正确！")

def main():
    parser = argparse.ArgumentParser(description='检查关键点数量是否匹配')
    parser.add_argument('--gt_file', type=str, default='/hdd/srt19/data/xray/inference_results/压缩性骨折全脊柱_images/gts.json', 
                      help='真实标注文件路径 (JSON格式)')
    parser.add_argument('--result_file', type=str, default='/hdd/srt19/data/xray/inference_results/压缩性骨折全脊柱_images/iou_results.json',
                      help='预测结果文件路径 (JSON格式)')
    
    args = parser.parse_args()
    check_keypoints(args.gt_file, args.result_file)

if __name__ == "__main__":
    main() 