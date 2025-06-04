import re
import json
import math
import pdb
from scipy.spatial import ConvexHull
import numpy as np
from mathruler.grader import grade_answer
# from .robopoint import robopoint_point_in_box_reward, robopoint_point_l1_reward, robopoint_point_format_reward


def embodied_r1_thinking_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0


def parse_ground_truth(ground_truth: str):
    """
    解析ground_truth，提取类型和实际答案
    格式为：<type>...</type>后跟实际答案
    """
    type_pattern = r'<type>(.*?)</type>'
    type_match = re.search(type_pattern, ground_truth, re.DOTALL)
    type_value = type_match.group(1).strip()
    # 提取type标签后的实际答案
    actual_answer = ground_truth[type_match.end():].strip()
    return type_value, actual_answer


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    计算预测答案与ground_truth的准确率
    """
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if grade_answer(given_answer, ground_truth):
            return 1.0

    except Exception:
        pass

    return 0.0

# 批量处理多个点的函数
def points_in_polygon_batch(points, polygon):
    """
    批量检查多个点是否在多边形内
    """
    results = []
    poly_array = np.array(polygon)
    
    # 如果多边形不闭合，则闭合它
    if not np.array_equal(poly_array[0], poly_array[-1]):
        poly_array = np.vstack([poly_array, poly_array[0]])
    
    for point in points:
        x, y = point
        
        # 计算射线与多边形边的交点
        inside = False
        j = len(poly_array) - 1
        
        for i in range(len(poly_array)):
            if ((poly_array[i, 1] > y) != (poly_array[j, 1] > y)) and \
               (x < poly_array[i, 0] + (poly_array[j, 0] - poly_array[i, 0]) * 
                (y - poly_array[i, 1]) / (poly_array[j, 1] - poly_array[i, 1])):
                inside = not inside
            j = i
            
        results.append(inside)
    
    return np.array(results)

def compute_discrete_frechet_distance(traj1, traj2):
    """
    计算两条轨迹之间的离散Fréchet距离
    traj1, traj2: 轨迹点列表，每个点是[x, y]格式
    """
    n = len(traj1)
    m = len(traj2)
    
    # 计算欧氏距离矩阵
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = math.sqrt((traj1[i][0] - traj2[j][0])**2 + (traj1[i][1] - traj2[j][1])**2)
    
    # 计算离散Fréchet距离的动态规划矩阵
    dp = np.zeros((n, m))
    dp[0, 0] = dist_matrix[0, 0]
    
    # 初始化第一行和第一列
    for i in range(1, n):
        dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
    
    # 填充DP表
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]),
                dist_matrix[i, j]
            )
    
    return dp[n-1, m-1]


def compute_hausdorff_distance(traj1, traj2):
    """
    计算两条轨迹之间的Hausdorff距离
    traj1, traj2: 轨迹点列表，每个点是[x, y]格式
    """
    n = len(traj1)
    m = len(traj2)
    
    # 计算从traj1到traj2的单向Hausdorff距离
    max_dist_1_to_2 = 0
    for point1 in traj1:
        min_dist = float('inf')
        for point2 in traj2:
            dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            min_dist = min(min_dist, dist)
        max_dist_1_to_2 = max(max_dist_1_to_2, min_dist)
    
    # 计算从traj2到traj1的单向Hausdorff距离
    max_dist_2_to_1 = 0
    for point2 in traj2:
        min_dist = float('inf')
        for point1 in traj1:
            dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            min_dist = min(min_dist, dist)
        max_dist_2_to_1 = max(max_dist_2_to_1, min_dist)
    
    # Hausdorff距离是两个单向距离的最大值
    return max(max_dist_1_to_2, max_dist_2_to_1)


def compute_rmse_with_interpolation(traj1, traj2):
    """
    使用线性插值计算两条轨迹的RMSE
    """
    # 将较短的轨迹插值到较长轨迹的长度
    target_len = max(len(traj1), len(traj2))
    
    # 转换为numpy数组便于插值
    traj1_np = np.array(traj1)
    traj2_np = np.array(traj2)
    
    # 生成插值的位置参数
    t1 = np.linspace(0, 1, len(traj1))
    t2 = np.linspace(0, 1, len(traj2))
    t_new = np.linspace(0, 1, target_len)
    
    # 对x和y坐标分别进行插值
    traj1_interp_x = np.interp(t_new, t1, traj1_np[:, 0])
    traj1_interp_y = np.interp(t_new, t1, traj1_np[:, 1])
    traj2_interp_x = np.interp(t_new, t2, traj2_np[:, 0])
    traj2_interp_y = np.interp(t_new, t2, traj2_np[:, 1])
    
    # 计算RMSE
    squared_error_sum = np.sum((traj1_interp_x - traj2_interp_x)**2 + 
                              (traj1_interp_y - traj2_interp_y)**2)
    rmse = np.sqrt(squared_error_sum / target_len)
    return rmse


def distance_to_reward(distance, min_threshold, max_threshold):
    """
    将距离转换为奖励值
    distance: 计算得到的距离
    min_threshold: 最小距离阈值，小于等于此值获得最大奖励
    max_threshold: 最大距离阈值，大于等于此值获得最小奖励
    """
    if distance <= min_threshold:
        return 1.0
    elif distance >= max_threshold:
        return 0.0
    else:
        return 1.0 - (distance - min_threshold) / (max_threshold - min_threshold)


def embodiedr1_nothinking_compute_score(predict_str: str, ground_truth: str) -> float:
    # 解析ground_truth中的type和实际答案
    type_value, actual_ground_truth = parse_ground_truth(ground_truth)

    # stage 1
    # virl: virl_general_qa
    # whatsup: spatial_qa
    # SAT: spatial_qa
    # stage 2
    # robopoint: point_ref
    # fsd: fsd_free_point
    # fsd: fsd_visual_trace

    if type_value == "point_ref":
        point_format_reward = robopoint_point_format_reward(predict_str)
        point_l1_reward = 0.0
        point_in_box_reward = 0.0
        if point_format_reward:
            ground_truth_points = json.loads(actual_ground_truth)

            try:
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                # print(f"point_content: {point_content}")
                predict_points = json.loads(point_content)

                assert isinstance(predict_points, list)
                # 检查列表中的元素是否都是列表，且长度为2
                for point in predict_points:
                    assert isinstance(point, list) and len(point) == 2
                
                # 解析成功，计算point l1 reward
                point_l1_reward = robopoint_point_l1_reward(predict_points, ground_truth_points)
                # 解析成功，计算point in box reward
                # 如果ground_truth_points小于3个点，则围不成多边形，无法计算点在框内得分，使用l1得分代替
                # 如果ground_truth_points大于等于3个点，但是无法构成凸包，则无法计算点在框内得分，使用l1得分代替
                if len(ground_truth_points) < 3:
                    point_in_box_reward = point_l1_reward
                else:
                    try:
                        point_in_box_reward = robopoint_point_in_box_reward(predict_points, ground_truth_points)
                    except Exception as e:
                        point_in_box_reward = point_l1_reward
                        # print(f"Error in robopoint_point_in_box_reward: {e}")
            except Exception as e:
                # print(f"Error parsing point content or calculating reward: {e} predict_str: {predict_str} ground_truth: {ground_truth}")
                pass
    
        overall_reward = \
            0.1*point_format_reward + \
            0.5*point_l1_reward + \
            0.3*point_in_box_reward
        return {
            "overall": overall_reward,
            "point_format": point_format_reward,
            "point_l1": point_l1_reward,
            "point_in_box": point_in_box_reward,
        }
    
    elif type_value == "fsd_free_point":
        # 预测答案需要包含在<think></think>和<answer></answer>标签中
        # 点的坐标需要使用<point>[[x1,y1],[x2,y2],...[xn,yn]]</point>格式提供
        # 预测的点必须是一个JSON格式的二维坐标点列表
        point_format_reward = robopoint_point_format_reward(predict_str)
        point_in_box_reward = 0.0
        point_distance_reward = 0.0
        
        if point_format_reward:
            try:
                # 解析ground truth，包含bounding box和8个关键点
                # {
                #   "bbox": [x1, y1, x2, y2],
                #   "free_points": [[x1,y1], [x2,y2], ..., [x8,y8]]
                # }
                gt_data = json.loads(actual_ground_truth)
                bbox = gt_data.get("bbox", [])
                key_points = gt_data.get("free_points", [])
                
                # 验证ground truth数据格式
                assert isinstance(bbox, list) and len(bbox) == 4  # [x1, y1, x2, y2]
                assert isinstance(key_points, list) and len(key_points) == 8
                for point in key_points:
                    assert isinstance(point, list) and len(point) == 2
                
                # 解析预测的点
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_points = json.loads(point_content)
                
                # 验证预测点的格式
                assert isinstance(predict_points, list)
                for point in predict_points:
                    assert isinstance(point, list) and len(point) == 2
                
                # 计算点在bbox内的比例
                points_in_box_count = 0
                for point in predict_points:
                    if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
                        points_in_box_count += 1
                
                # 点在bbox内的奖励
                if len(predict_points) > 0:
                    point_in_box_reward = points_in_box_count / len(predict_points)
                
                # 计算预测点与关键点的平均距离
                # 使用robopoint_point_l1_reward函数计算距离奖励
                point_distance_reward = robopoint_point_l1_reward(predict_points, key_points)
            
            except Exception as e:
                #print(f"Error parsing content or calculating reward: {e}, predict_str: {predict_str}, ground_truth: {ground_truth}")
                pass
        
        overall_reward = \
            0.1*point_format_reward + \
            0.6*point_in_box_reward + \
            0.3*point_distance_reward
            
        return {
            "overall": overall_reward,
            "point_format": point_format_reward,
            "point_in_box": point_in_box_reward,
            "point_distance": point_distance_reward,
        }
    elif type_value == "fsd_visual_trace":
        # 预测答案需要包含在<think></think>和<answer></answer>标签中
        # 轨迹点的坐标需要使用<point>[[x1,y1],[x2,y2],...[xn,yn]]</point>格式提供
        point_format_reward = robopoint_point_format_reward(predict_str)
        frechet_reward = 0.0
        rmse_reward = 0.0
        point_num_reward = 0.0
        
        if point_format_reward:
            try:
                # 解析ground truth，包含参考轨迹
                # {
                #   "trajectory": [[x1,y1], [x2,y2], ..., [xn,yn]]
                # }
                gt_data = json.loads(actual_ground_truth)
                gt_trajectory = gt_data.get("trajectory", [])
                
                # 验证ground truth数据格式
                assert isinstance(gt_trajectory, list) and len(gt_trajectory) > 0
                for point in gt_trajectory:
                    assert isinstance(point, list) and len(point) == 2
                
                # 解析预测的轨迹点
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_trajectory = json.loads(point_content)
                
                # 验证预测轨迹点的格式
                assert isinstance(predict_trajectory, list) and len(predict_trajectory) > 0
                for point in predict_trajectory:
                    assert isinstance(point, list) and len(point) == 2
                
                # 检查是否正好有8个点，只有满足条件才计算距离奖励
                if len(predict_trajectory) == 8:
                    point_num_reward = 1.0
                    
                    # 计算离散Fréchet距离
                    dfd = compute_discrete_frechet_distance(predict_trajectory, gt_trajectory)
                    frechet_reward = distance_to_reward(dfd, 10, 100)
                    
                    # 计算RMSE
                    rmse = compute_rmse_with_interpolation(predict_trajectory, gt_trajectory)
                    rmse_reward = distance_to_reward(rmse, 5, 50)
            
            except Exception as e:
                #print(f"Error parsing content or calculating reward: {e}, predict_str: {predict_str}, ground_truth: {ground_truth}")
                pass
        
        overall_reward = \
            0.1*point_format_reward + \
            0.4*frechet_reward + \
            0.4*rmse_reward + \
            0.1*point_num_reward
            
        return {
            "overall": overall_reward,
            "point_format": point_format_reward,
            "point_num": point_num_reward,
            "frechet_distance": frechet_reward,
            "rmse_distance": rmse_reward,
        }

    
    else:
        raise ValueError(f"Unknown type: {type_value}")

    