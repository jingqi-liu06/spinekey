import json
import numpy as np
from shapely.geometry import Polygon

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_iou(poly1_coords, poly2_coords):
    # 将坐标转换为Shapely多边形对象
    poly1 = Polygon(poly1_coords)
    poly2 = Polygon(poly2_coords)
    
    # 计算交集和并集面积
    try:
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        # 计算IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
    except:
        return 0

def process_annotations(gts_path, results_path, output_path, iou_threshold=0.5):
    # 加载JSON文件
    gts = load_json(gts_path)
    results = load_json(results_path)
    
    # 创建输出结果字典
    output_data = {}
    
    # 处理每个图片
    for img_name in gts.keys():
        if img_name not in results:
            continue
            
        output_data[img_name] = {}
        gt_regions = gts[img_name]
        result_regions = results[img_name]
        
        # 对于每个真实标注区域
        for gt_name, gt_coords in gt_regions.items():
            best_iou = 0
            best_match_name = None
            best_match_coords = None
            
            # 与每个预测区域计算IoU
            for result_name, result_coords in result_regions.items():
                # 跳过无效坐标（全是[1,1]的情况）
                if all(x == 1 for coord in result_coords for x in coord):
                    continue
                    
                iou = calculate_iou(gt_coords, result_coords)
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_name = result_name
                    best_match_coords = result_coords
            
            # 如果找到匹配的预测框
            if best_match_coords is not None:
                output_data[img_name][gt_name] = best_match_coords
    
    # 保存结果到新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    gts_path = "/home/ljq/data_xray/inference_data/iou_results/疑似骨折全脊柱_images/gts.json"
    results_path = "/home/ljq/data_xray/inference_data/iou_results/疑似骨折全脊柱_images/results.json"
    output_path = "/home/ljq/data_xray/inference_data/iou_results/疑似骨折全脊柱_images/iou_results.json"
    process_annotations(gts_path, results_path, output_path)
