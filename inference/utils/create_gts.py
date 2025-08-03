import json
from tqdm import tqdm

def create_gts(anns_path, gts_path):
    """
    从原始标注文件生成ground truth文件
    
    Args:
        anns_path: 输入的标注文件路径
        gts_path: 输出的ground truth文件路径
    """
    with open(anns_path, "r") as f:
        anns = json.load(f)
        
    gts_dict = {}
    for ann in tqdm(anns, desc="处理标注"):
        gt = {}

        # 处理四边形
        quads = ann["quads"]
        for quad_name in quads:
            quad = [(quads[quad_name][i*2], quads[quad_name][i*2+1]) for i in range(4)]
            gt[quad_name] = quad

        # 处理三角形
        triangles = ann["triangles"]
        for triangle_name in triangles:
            triangle = [(triangles[triangle_name][i*2], triangles[triangle_name][i*2+1]) for i in range(3)]
            gt[triangle_name] = triangle

        # 处理圆形
        circles = ann["circles"]
        for circle_name in circles:
            circle = (circles[circle_name][0], circles[circle_name][1], circles[circle_name][2])
            gt[circle_name] = circle

        filename = ann["filename"]
        gts_dict[filename] = gt

    with open(gts_path, "w") as f:
        json.dump(gts_dict, f)
        
    return gts_dict

if __name__ == "__main__":
    # 当作为独立脚本运行时的示例用法
    anns_path = "/home/shiym/datasets/vertebra-det/00_neibushuju/neibushuju_sagittal.json"
    gts_path = "/home/shiym/datasets_processed/vertebra-det/sagittal/gts.json"
    create_gts(anns_path, gts_path)