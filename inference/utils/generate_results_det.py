import argparse
import json
import math
import os
import os.path as osp
from tqdm import tqdm
import time

import cv2
import imutils
import numpy as np
from mmdet.apis import init_detector, inference_detector

obj_names = [
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "S1",
]


def postprocess_det(result):
    result = result.cpu()
    scores = np.array(result.pred_instances.scores)
    bboxes = np.array(result.pred_instances.bboxes)
    masks = np.array(result.pred_instances.masks)
    classes = np.array(result.pred_instances.labels)
    obj_count = np.sum(scores >= 0.3) if np.sum(scores >= 0.3) <= 24 else 24
    if obj_count == 0:
        return None, None

    # sort by scores
    sorted_indices = np.argsort(-scores)
    scores = scores[sorted_indices]
    bboxes = bboxes[sorted_indices]
    masks = masks[sorted_indices]
    classes = classes[sorted_indices]

    # filter by scores
    scores = scores[:obj_count]
    bboxes = bboxes[:obj_count]
    masks = masks[:obj_count]
    classes = classes[:obj_count]

    # sort by bboxes
    sorted_indices = np.argsort(bboxes[:, 1] + bboxes[:, 3])
    scores = scores[sorted_indices]
    bboxes = bboxes[sorted_indices]
    masks = masks[sorted_indices]
    classes = classes[sorted_indices]
    # soluton 1： maskrcnn 输出 分类+定位 
    # TODO：2. 重写生成json函数和绘图函数; 1. 重新训练model
    # transform
    bboxes = np.concatenate((bboxes, scores.reshape(-1, 1)), axis=1)
    segs = []
    for idx in range(obj_count):
        segs.append(masks[idx])

    # split result
    result_24 = ([bboxes[: obj_count]], [segs[: obj_count]])
    return result_24, None


def clockwiseangle_and_distance(origin, point):
    refvec = [1, 0]
    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    # if angle < 0:
    #     return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


# 保证点是顺时针排列，第一个点位于左上
def Clockwise(pts):
    """
    pts是一维array形式，包含4个点
    """
    pts = np.array(pts).reshape((-1, 2))

    # 根据x进行排序
    pts = pts[pts[:, 0].argsort()]
    left_two_points = pts[:2].copy()
    right_two_points = pts[2:].copy()

    # 根据y进行排序
    left_two_points = left_two_points[left_two_points[:, 1].argsort()]
    origin = left_two_points[0]

    rest = np.concatenate([right_two_points, left_two_points[[1]]], axis=0)

    angle = []
    for box in rest:
        angle.append(clockwiseangle_and_distance(origin, box)[0])
    angle = np.array(angle)
    order = np.array(angle).argsort()[::-1]
    rest = rest[order]
    rect = np.vstack((origin, rest))
    # return the ordered coordinates
    return np.reshape(rect, -1).tolist()


def convert_one_spine(mask):
    # 转换成灰度图
    gray = np.array(255 * np.array(mask, dtype=np.int8), dtype="uint8")

    # 轮廓检测
    contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    # 有多个轮廓 每个轮廓由多个点组成 选择点数最多的轮廓
    contour_lens = [len(contour) for contour in contours]
    max_len_index = np.array(contour_lens).argmax()
    contour = contours[max_len_index]

    # 轮廓的最小外接矩形 rect((x,y),(w,h),rotation) 中心坐标 长 宽 旋转角度
    rect = cv2.minAreaRect(contour)

    # 外接矩形4个顶点的坐标
    points_of_rect = np.int64(cv2.boxPoints(rect))
    points_of_rect = np.reshape(Clockwise(points_of_rect), (-1, 2))

    # 对于外接矩形的每个点，在轮廓的所有点中 选择k个与之距离最近的点 这些点的中心作为外接矩形点的坐标。
    contour = np.reshape(contour, (-1, 2))
    for i, point_of_rect in enumerate(points_of_rect):
        dis = point_of_rect - contour
        square_dis = (dis[:, 0]) ** 2 + (dis[:, 1]) ** 2
        k = 6
        min_k_index = square_dis.argsort()[0:k]
        min_k = contour[min_k_index]
        points_of_rect[i] = np.array(np.sum(min_k, axis=0) / k, dtype="int32")

    points_of_rect = points_of_rect.reshape(-1).tolist()

    return points_of_rect


def convert_spine_of_one_img(model_out, max_count=24):
    if model_out is None:
        return {obj_name: [(1, 1)] * 4 for obj_name in obj_names[:-1]}

    spines = {}
    masks = model_out[1][0]
    max_count = min(max_count, len(masks))
    masks = masks[:max_count]
    for index, mask in enumerate(masks):
        # print(index)
        pts = convert_one_spine(mask)
        # spines.append(pts)
        pts_tuple = [(pts[i * 2], pts[i * 2 + 1]) for i in range(4)]
        spines[obj_names[index]] = pts_tuple

    # 补全缺失的椎骨
    for obj_name in obj_names[:-1]:
        if obj_name not in spines:
            spines[obj_name] = [(1, 1)] * 4

    return spines


def convert_one_s1(mask):
    # 转换成灰度图
    gray = np.array(255 * np.array(mask, dtype=np.int8), dtype="uint8")

    # 轮廓检测
    contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    # 有多个轮廓 每个轮廓由多个点组成 选择点数最多的轮廓
    contour_lens = [len(contour) for contour in contours]
    max_len_index = np.array(contour_lens).argmax()
    contour = contours[max_len_index]
    # contour=np.reshape(contour,(-1,2))

    # 最小面积矩形 rect((x,y),(w,h),rotation) 中心坐标 长 宽 旋转角度
    triangle = cv2.minEnclosingTriangle(contour)

    # 外接三角形3个顶点的坐标
    points_of_triangle = np.int64(triangle[1])
    points_of_triangle = np.reshape(Clockwise(points_of_triangle), (-1, 2))
    points_of_triangle = points_of_triangle.reshape(-1).tolist()

    return points_of_triangle


def convert_s1_of_one_img(model_out, max_count=1):
    if model_out is None:
        return {obj_names[-1]: [(1, 1)] * 3}

    pts = []
    masks = model_out[1][0]
    if len(masks) == 0:
        return pts

    max_count = min(max_count, len(masks))
    masks = masks[:max_count]

    mask = masks[0]
    pts = convert_one_s1(mask)
    pts_tuple = [(pts[i * 2], pts[i * 2 + 1]) for i in range(3)]

    return {obj_names[-1]: pts_tuple}


def detect_vertebrae_det(images_dir, cfg_det_path, ckpt_det_path, results_path, cuda_id=0, max_count=24):
    model = init_detector(
        cfg_det_path,
        ckpt_det_path,
        device=f"cuda:{cuda_id}",
    )

    total_time = 0
    total_frames = 0
    
    results = {}
    for filename in tqdm(os.listdir(images_dir)):
        start_time = time.time()
        
        # 包含所有处理步骤
        result = inference_detector(model, osp.join(images_dir, filename))
        result_24, result_S1 = postprocess_det(result)
        result_dict = {}
        result_dict.update(convert_spine_of_one_img(result_24))
        result_dict.update(convert_s1_of_one_img(result_S1))
        results.update({filename: result_dict})
        
        end_time = time.time()
        total_time += (end_time - start_time)
        total_frames += 1

    avg_fps = total_frames / total_time if total_time > 0 else 0
    
    fps_path = os.path.join(os.path.dirname(results_path), 'inference_speed.txt')
    with open(fps_path, 'w') as f:
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Average inference speed: {avg_fps:.2f} FPS")

    os.makedirs(osp.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f)

    return results



