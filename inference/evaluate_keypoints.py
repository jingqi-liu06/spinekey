from utils.keypoint_metrics import KeypointMetrics
import argparse

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='评估关键点检测结果')
    parser.add_argument('--gt_file', type=str, default='gts.json', help='真实标注文件路径 (JSON格式)')
    parser.add_argument('--result_file', type=str, default='iou_results.json', help='预测结果文件路径 (JSON格式)，命名格式应该是iou_results.json')
    parser.add_argument('--output', type=str, default='evaluation_results.txt',
                      help='输出结果文件路径 (默认: evaluation_results.txt)')
    
    # 解析命令行参数
    args = parser.parse_args()

    # 初始化KeypointMetrics类
    metrics = KeypointMetrics(args.gt_file, args.result_file)

    # 计算并获取指标
    results = metrics.evaluate()

    # 输出结果到控制台和文件
    with open(args.output, 'w', encoding='utf-8') as f:
        for metric, value in results.items():
            # 打印到控制台
            print(f"{metric}: {value}")
            # 写入到文件
            f.write(f"{metric}: {value}\n")
    
    print(f"\n结果已保存到 {args.output}")

if __name__ == "__main__":
    main() 