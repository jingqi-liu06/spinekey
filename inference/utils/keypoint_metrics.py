import json
import numpy as np

class KeypointMetrics:
    def __init__(self, gt_file, pred_file):
        with open(gt_file, 'r') as f:
            self.ground_truths = json.load(f)
        
        with open(pred_file, 'r') as f:
            self.predictions = json.load(f)

    def calculate_metrics(self, gt, pred):
        # 计算欧几里得距离
        distances = np.linalg.norm(np.array(gt) - np.array(pred), axis=1)
        
        # 计算图像对角线长度和阈值
        image_width = 2560
        image_height = 1440
        diagonal_length = np.sqrt(image_width**2 + image_height**2)
        threshold = 0.001 * diagonal_length  # 使用图像对角线长度1%作为阈值
        
        # 计算准确度
        correct_predictions = np.sum(distances < threshold)
        accuracy = correct_predictions / len(distances) * 100
        
        # 计算AP（Average Precision）
        precisions = []
        recalls = []
        thresholds = np.linspace(0, threshold * 2, 100)  # 使用多个阈值计算PR曲线
        
        for t in thresholds:
            tp = np.sum(distances < t)
            fp = np.sum(distances >= t)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(distances)
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算AP（使用11点插值法）
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                ap += np.max(np.array(precisions)[mask]) / 11
        
        # 平均欧几里得距离
        avg_euclidean_distance = np.mean(distances)
        
        # 正确关键点百分比
        percentage_of_correct_keypoints = correct_predictions / len(gt) * 100
        
        # 均方根误差
        rmse = np.sqrt(np.mean(distances ** 2))
        
        # 平均绝对误差
        mae = np.mean(np.abs(distances))
        
        # 正确关键点百分比（PCK）
        pck = percentage_of_correct_keypoints
        
        return avg_euclidean_distance, percentage_of_correct_keypoints, rmse, mae, pck, accuracy, ap

    def evaluate(self):
        total_avg_euclidean_distance = 0
        total_percentage_of_correct_keypoints = 0
        total_rmse = 0
        total_mae = 0
        total_pck = 0
        total_accuracy = 0
        total_ap = 0
        num_images = 0

        for image_id, gt_keypoints in self.ground_truths.items():
            if image_id in self.predictions:
                pred_keypoints = self.predictions[image_id]
                for key in gt_keypoints:
                    if key in pred_keypoints:
                        gt = gt_keypoints[key]
                        pred = pred_keypoints[key]
                        metrics = self.calculate_metrics(gt, pred)
                        total_avg_euclidean_distance += metrics[0]
                        total_percentage_of_correct_keypoints += metrics[1]
                        total_rmse += metrics[2]
                        total_mae += metrics[3]
                        total_pck += metrics[4]
                        total_accuracy += metrics[5]
                        total_ap += metrics[6]
                        num_images += 1

        # 计算平均值
        average_accuracy = total_accuracy / num_images
        average_map = total_ap / num_images  # mAP是所有图像AP的平均值

        return {
            "平均欧几里得距离(Average Euclidean Distance)": total_avg_euclidean_distance / num_images,
            "正确关键点百分比(PCK)": total_percentage_of_correct_keypoints / num_images,
            "均方根误差(RMSE)": total_rmse / num_images,
            "平均绝对误差(MAE)": total_mae / num_images,
            "准确度(Accuracy)": average_accuracy,
            "mAP": average_map
        } 