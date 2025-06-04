import re
import json
import math
import numpy as np
from mathruler.grader import grade_answer
from scipy.spatial import ConvexHull


def embodied_r1_thinking_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0


def embodied_r1_point_format_reward(predict_str: str) -> float:
    pattern = r"<point>.*?</point>"
    match = re.search(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0


def embodied_r1_3d_point_format_reward(predict_str: str) -> float:
    """
    检查预测字符串是否同时包含正确格式的point和depth标签
    point格式: <point>[[x1,y1],[x2,y2],...,[xn,yn]]</point>
    depth格式: <depth>[depth1,depth2,...,depthn]</depth>
    
    Args:
        predict_str: 预测的字符串
        
    Returns:
        如果两个标签都存在且格式正确返回1.0，否则返回0.0
    """
    if not isinstance(predict_str, str):
        return 0.0
        
    try:
        # 检查point标签
        point_pattern = r"<point>(.*?)</point>"
        point_match = re.search(point_pattern, predict_str, re.DOTALL)
        if not point_match:
            return 0.0
            
        # 检查depth标签
        depth_pattern = r"<depth>(.*?)</depth>"
        depth_match = re.search(depth_pattern, predict_str, re.DOTALL)
        if not depth_match:
            return 0.0
            
        try:
            # 验证point内容是否为有效的JSON数组
            point_content = point_match.group(1).strip()
            point_data = json.loads(point_content)
            if not isinstance(point_data, list):
                return 0.0
                
            for point in point_data:
                if not isinstance(point, list) or len(point) != 2:
                    return 0.0
                # 确保坐标是数值类型
                if not all(isinstance(coord, (int, float)) for coord in point):
                    return 0.0
        except:
            return 0.0
            
        try:
            # 验证depth内容是否为有效的JSON数组
            depth_content = depth_match.group(1).strip()
            depth_data = json.loads(depth_content)
            if not isinstance(depth_data, list):
                return 0.0
                
            # 确保深度值是数值类型
            if not all(isinstance(d, (int, float)) for d in depth_data):
                return 0.0
                
            # 确保point和depth数组长度相同
            if len(point_data) != len(depth_data):
                return 0.0
        except:
            return 0.0
            
        return 1.0
        
    except:
        return 0.0


def parse_ground_truth(ground_truth: str):
    """
    Parse ground_truth to extract type and actual answer.
    Format: <type>...</type> followed by actual answer
    """
    type_pattern = r'<type>(.*?)</type>'
    type_match = re.search(type_pattern, ground_truth, re.DOTALL)
    type_value = type_match.group(1).strip()
    # Extract actual answer after type tag
    actual_answer = ground_truth[type_match.end():].strip()
    return type_value, actual_answer


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    Calculate accuracy between predicted answer and ground_truth
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


def robopoint_point_in_box_reward(predict_points: list, ground_truth_points: list) -> float:
    """
    Check if predicted points are inside the polygon formed by ground_truth points.
    If ground_truth_points has less than 3 points, cannot form a polygon, use L1 score instead.
    If ground_truth_points has 3 or more points but cannot form a convex hull, use L1 score instead.
    """
    assert isinstance(predict_points, list)
    assert isinstance(ground_truth_points, list)
    assert len(ground_truth_points) >= 3

    # Build convex hull
    points = np.array(ground_truth_points)
    
    hull = ConvexHull(points)
    
    # Check if point is inside convex hull
    def in_hull(point, hull):
        return all((np.dot(eq[:-1], point) + eq[-1] <= 0) for eq in hull.equations)
    
    all_points_in_hull = True
    for predict_point in predict_points:
        if not in_hull(predict_point, hull):
            all_points_in_hull = False
            break
    
    if all_points_in_hull:
        return 1.0
    
    return 0.0


def robopoint_point_l1_reward(predict_points: list, ground_truth_points: list) -> float:
    try:
        assert isinstance(predict_points, list)
        assert isinstance(ground_truth_points, list)

        # Return 0 if predict_points is empty
        if len(predict_points) == 0:
            return 0.0
            
        predict_array = np.array(predict_points)
        gt_array = np.array(ground_truth_points)
        
        # Calculate distances between each predict point and ground truth points
        # Shape: (num_predict_points, num_gt_points) 
        distances = np.sqrt(np.sum((predict_array[:, np.newaxis, :] - gt_array[np.newaxis, :, :]) ** 2, axis=2))
        
        min_distances = np.min(distances, axis=1)
        average_min_distance = np.mean(min_distances)
        
        MIN_DISTANCE_THRESHOLD = 10
        MAX_DISTANCE_THRESHOLD = 50
        
        if average_min_distance <= MIN_DISTANCE_THRESHOLD:
            reward = 1.0
        elif average_min_distance >= MAX_DISTANCE_THRESHOLD:
            reward = 0.0
        else:
            reward = 1.0 - (average_min_distance - MIN_DISTANCE_THRESHOLD) / (MAX_DISTANCE_THRESHOLD - MIN_DISTANCE_THRESHOLD)
        
        return reward
    except Exception as e:
        return 0.0


def points_in_polygon_batch(points, polygon):
    """
    Batch check if multiple points are inside a polygon
    """
    results = []
    poly_array = np.array(polygon)
    
    # Close the polygon if not closed
    if not np.array_equal(poly_array[0], poly_array[-1]):
        poly_array = np.vstack([poly_array, poly_array[0]])
    
    for point in points:
        x, y = point
        
        # Calculate intersection points between ray and polygon edges
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
    Calculate discrete Fréchet distance between two trajectories
    traj1, traj2: trajectory point lists, each point in [x, y] format
    """
    n = len(traj1)
    m = len(traj2)
    
    # Calculate Euclidean distance matrix
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = math.sqrt((traj1[i][0] - traj2[j][0])**2 + (traj1[i][1] - traj2[j][1])**2)
    
    # Calculate dynamic programming matrix for discrete Fréchet distance
    dp = np.zeros((n, m))
    dp[0, 0] = dist_matrix[0, 0]
    
    # Initialize first row and column
    for i in range(1, n):
        dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
    
    # Fill DP table
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]),
                dist_matrix[i, j]
            )
    
    return dp[n-1, m-1]


def compute_rmse_with_interpolation(traj1, traj2):
    """
    Calculate RMSE between two trajectories using linear interpolation
    """
    # Interpolate shorter trajectory to match length of longer one
    target_len = max(len(traj1), len(traj2))
    
    # Convert to numpy arrays for interpolation
    traj1_np = np.array(traj1)
    traj2_np = np.array(traj2)
    
    # Generate interpolation position parameters
    t1 = np.linspace(0, 1, len(traj1))
    t2 = np.linspace(0, 1, len(traj2))
    t_new = np.linspace(0, 1, target_len)
    
    # Interpolate x and y coordinates separately
    traj1_interp_x = np.interp(t_new, t1, traj1_np[:, 0])
    traj1_interp_y = np.interp(t_new, t1, traj1_np[:, 1])
    traj2_interp_x = np.interp(t_new, t2, traj2_np[:, 0])
    traj2_interp_y = np.interp(t_new, t2, traj2_np[:, 1])
    
    # Calculate RMSE
    squared_error_sum = np.sum((traj1_interp_x - traj2_interp_x)**2 + 
                              (traj1_interp_y - traj2_interp_y)**2)
    rmse = np.sqrt(squared_error_sum / target_len)
    return rmse


def pixel_to_world(x, y, dep, img_width=720, img_height=480, camera_view_matrix_inv=np.array([[ -1  ,    0    ,     0 ,        0.5        ],[ 0.    ,     0.70710678   ,-0.70710678 , 0.7        ],[ 0.    ,     -0.70710678  ,-0.70710678 ,  1       ],[ 0,0,0 ,    1.        ]]).T, camera_proj_matrix=np.array([[1.7320507, 0., 0., 0.],[0., 2.5980759, 0., 0.],[0., 0., 0., -1.],[0., 0., 0.05, 0.]])):
    # 将矩阵转换为 Tensor
    # 计算主点坐标（图像中心）
    centerU = 360.0  # 
    centerV = 240.0 # 

    # 直接使用标量计算焦距参数
    fu = 1.1547  # 对应原始代码中的Tensor值
    fv = 0.7698

    # 计算相机坐标系下的坐标（纯NumPy运算）
    X_cam = (x - centerU) / img_width * dep * fu
    Y_cam = (y - centerV) / img_height * dep * fv
    Z_cam = dep

    # 构造齐次坐标点
    point_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])

    # 通过视图矩阵的逆变换到世界坐标系
    point_world = point_cam @ camera_view_matrix_inv

    # 返回XYZ坐标（转换为Python列表）
    return point_world[:3].tolist()


def evaluate_posi(tar_pos, mode, sel_pos=None, sel_pos_1=None, sel_pos_2=None, sel_pos_all=None):
    succ = 0
    if mode in ["left", "right", "front", "back", "behind", "top"]:
        if mode == "left":
            succ += sel_pos[1] > tar_pos[1]
        elif mode == "right":
            succ += sel_pos[1] < tar_pos[1]
        elif mode == "front":
            succ += sel_pos[0] < tar_pos[0]
        elif mode == "back" or mode == "behind":
            succ += sel_pos[0] > tar_pos[0]
        elif mode == "top":
            succ += sel_pos[2] <= tar_pos[2]
    elif mode == "between":
        max_sel_pos_x = np.max([sel_pos_1[0], sel_pos_2[0]])
        max_sel_pos_y = np.max([sel_pos_1[1], sel_pos_2[1]])
        min_sel_pos_x = np.min([sel_pos_1[0], sel_pos_2[0]])
        min_sel_pos_y = np.min([sel_pos_1[1], sel_pos_2[1]])
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) or (min_sel_pos_y < tar_pos[0] < max_sel_pos_y)
    elif mode == "center":
        max_sel_pos_x = np.max(sel_pos_all, axis=0)[0]
        min_sel_pos_x = np.min(sel_pos_all, axis=0)[0]
        max_sel_pos_y = np.max(sel_pos_all, axis=0)[1]
        min_sel_pos_y = np.min(sel_pos_all, axis=0)[1]
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) and (min_sel_pos_y < tar_pos[1] < max_sel_pos_y)
    return succ


def calculate_xy_depth_reward(object:list,direction:str,predict_answer:list)->float:
    real_pos = predict_answer
    if direction in ["left", "right", "front", "behind", "top"]:
            success = evaluate_posi(real_pos, direction, object[0])
    elif direction == "between":
            sel_pos_1 = object[0]
            sel_pos_2 = object[1]
            success = evaluate_posi(real_pos, direction, sel_pos_1=sel_pos_1, sel_pos_2=sel_pos_2)
    elif direction == "center":
            success = evaluate_posi(real_pos, direction, sel_pos_all=object)
    else:
        return 0.0

    return success


def distance_to_reward(distance, min_threshold, max_threshold):
    """
    Convert distance to reward value
    distance: calculated distance
    min_threshold: minimum distance threshold, gets maximum reward if less than or equal to this value
    max_threshold: maximum distance threshold, gets minimum reward if greater than or equal to this value
    """
    if distance <= min_threshold:
        return 1.0
    elif distance >= max_threshold:
        return 0.0
    else:
        return 1.0 - (distance - min_threshold) / (max_threshold - min_threshold)


def embodiedr1_compute_score(predict_str: str, ground_truth: str) -> float:
    # Parse type and actual answer from ground_truth
    type_value, actual_ground_truth = parse_ground_truth(ground_truth)

    # Calculate thinking format reward
    thinking_format_reward = embodied_r1_thinking_format_reward(predict_str)
    # stage 1
    # virl: virl_general_qa
    # whatsup: spatial_qa
    # SAT: spatial_qa
    # stage 2
    # robopoint: point_ref
    # fsd: fsd_free_point
    # fsd: fsd_visual_trace
    # roborefit: point_rec
    # refcoco: point_rec
    # handal: grounding_rec
    if type_value == "general_qa":
        accuracy = accuracy_reward(predict_str, actual_ground_truth)
        return {
            "overall": 0.1*thinking_format_reward + 0.9*accuracy,
            "thinking_format": thinking_format_reward,
            "general_qa_accuracy": accuracy,
        }
    elif type_value == "spatial_qa":
        accuracy = accuracy_reward(predict_str, actual_ground_truth)
        return {
            "overall": 0.1*thinking_format_reward + 0.9*accuracy,
            "thinking_format": thinking_format_reward,
            "spatial_qa_accuracy": accuracy,
        }
    elif type_value == "point_ref":
        point_format_reward = embodied_r1_point_format_reward(predict_str)
        point_l1_reward = 0.0
        point_in_box_reward = 0.0
        if point_format_reward:
            ground_truth_points = json.loads(actual_ground_truth)

            try:
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_points = json.loads(point_content)

                assert isinstance(predict_points, list)
                for point in predict_points:
                    assert isinstance(point, list) and len(point) == 2
                
                point_l1_reward = robopoint_point_l1_reward(predict_points, ground_truth_points)

                # if len(ground_truth_points) < 3:
                #     point_in_box_reward = point_l1_reward
                # else:
                #     try:
                #         point_in_box_reward = robopoint_point_in_box_reward(predict_points, ground_truth_points)
                #     except Exception as e:
                #         point_in_box_reward = point_l1_reward
            except Exception as e:
                pass
    
        overall_reward = 0.1*thinking_format_reward + \
            0.1*point_format_reward + \
            0.8*point_l1_reward
            # 0.5*point_in_box_reward
        return {
            "overall": overall_reward,
            "thinking_format": thinking_format_reward,
            "point_format": point_format_reward,
            "robopoint_point_distance": point_l1_reward,
            # "robopoint_point_in_box": point_in_box_reward,
        }
    elif type_value == "fsd_free_point":
        point_format_reward = embodied_r1_point_format_reward(predict_str)
        point_in_box_reward = 0.0
        point_distance_reward = 0.0
        
        if point_format_reward:
            try:
                # parse ground truth
                # {
                #   "bbox": [x1, y1, x2, y2],
                #   "free_points": [[x1,y1], [x2,y2], ..., [x8,y8]]
                # }
                gt_data = json.loads(actual_ground_truth)
                bbox = gt_data.get("bbox", [])
                key_points = gt_data.get("free_points", [])
                
                assert isinstance(bbox, list) and len(bbox) == 4
                assert isinstance(key_points, list) and len(key_points) == 8
                for point in key_points:
                    assert isinstance(point, list) and len(point) == 2
                
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_points = json.loads(point_content)
                
                assert isinstance(predict_points, list)
                for point in predict_points:
                    assert isinstance(point, list) and len(point) == 2
                
                points_in_box_count = 0
                for point in predict_points:
                    if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
                        points_in_box_count += 1
                
                if len(predict_points) > 0:
                    point_in_box_reward = points_in_box_count / len(predict_points)
                
                point_distance_reward = robopoint_point_l1_reward(predict_points, key_points)
            
            except Exception as e:
                pass
        
        overall_reward = 0.1*thinking_format_reward + \
            0.1*point_format_reward + \
            0.6*point_in_box_reward + \
            0.2*point_distance_reward
            
        return {
            "overall": overall_reward,
            "thinking_format": thinking_format_reward,
            "point_format": point_format_reward,
            "fsd_point_in_box": point_in_box_reward,
            "fsd_point_distance": point_distance_reward,
        }
    elif type_value == "fsd_visual_trace":
        point_format_reward = embodied_r1_point_format_reward(predict_str)
        frechet_reward = 0.0
        rmse_reward = 0.0
        point_num_reward = 0.0
        
        if point_format_reward:
            try:
                # parse ground truth, contains reference trajectory
                # {
                #   "trajectory": [[x1,y1], [x2,y2], ..., [xn,yn]]
                # }
                gt_data = json.loads(actual_ground_truth)
                gt_trajectory = gt_data.get("trajectory", [])
                
                assert isinstance(gt_trajectory, list) and len(gt_trajectory) > 0
                for point in gt_trajectory:
                    assert isinstance(point, list) and len(point) == 2
                
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_trajectory = json.loads(point_content)
                
                assert isinstance(predict_trajectory, list) and len(predict_trajectory) > 0
                for point in predict_trajectory:
                    assert isinstance(point, list) and len(point) == 2
                
                if len(predict_trajectory) == 8:
                    point_num_reward = 1.0
                    
                    dfd = compute_discrete_frechet_distance(predict_trajectory, gt_trajectory)
                    frechet_reward = distance_to_reward(dfd, 10, 100)
                    
                    rmse = compute_rmse_with_interpolation(predict_trajectory, gt_trajectory)
                    rmse_reward = distance_to_reward(rmse, 5, 50)
            
            except Exception as e:
                pass
        
        overall_reward = 0.1*thinking_format_reward + \
            0.1*point_format_reward + \
            0.1*point_num_reward + \
            0.3*frechet_reward + \
            0.4*rmse_reward
            
        return {
            "overall": overall_reward,
            "thinking_format": thinking_format_reward,
            "point_format": point_format_reward,
            "fsd_point_num": point_num_reward,
            "fsd_frechet_distance": frechet_reward,
            "fsd_rmse_distance": rmse_reward,
        }
    elif type_value == "point_rec":
        point_format_reward = embodied_r1_point_format_reward(predict_str)
        point_in_box_reward = 0.0
        if point_format_reward:
            
            try:
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_points = json.loads(point_content)
                
                assert isinstance(predict_points, list)
                for point in predict_points:
                    assert isinstance(point, list) and len(point) == 2
                
                # parse ground truth
                # {
                #   "segmentation": [[x1,y1], [x2,y2], ..., [x8,y8]]
                # }
                ground_truth_points = json.loads(actual_ground_truth)
                segmentation = ground_truth_points["segmentation"]
                assert isinstance(segmentation, list) and len(segmentation) >= 3
                
                results = points_in_polygon_batch(predict_points, segmentation)
                points_in_region_count = np.sum(results)
                
                if len(predict_points) > 0:
                    point_in_box_reward = points_in_region_count / len(predict_points)
                
            except Exception as e:
                pass
        
        overall_reward = 0.1*thinking_format_reward + \
            0.1*point_format_reward + \
            0.8*point_in_box_reward
            
        return {
            "overall": overall_reward,
            "thinking_format": thinking_format_reward,
            "point_format": point_format_reward,
            "refcoco_point_in_mask": point_in_box_reward,
        }
    elif type_value == "grounding_rec":
        point_format_reward = embodied_r1_point_format_reward(predict_str)
        point_in_box_reward = 0.0
        if point_format_reward:
            try:
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_points = json.loads(point_content)
                
                assert isinstance(predict_points, list)
                for point in predict_points:
                    assert isinstance(point, list) and len(point) == 2
                
                # parse ground truth, contains segmentation
                # {
                #   "segmentation": [[x1,y1], [x2,y2], ..., [x8,y8]]
                # }
                ground_truth_points = json.loads(actual_ground_truth)
                segmentation = ground_truth_points["segmentation"]
                assert isinstance(segmentation, list) and len(segmentation) >= 3
                
                results = points_in_polygon_batch(predict_points, segmentation)
                points_in_region_count = np.sum(results)
                
                if len(predict_points) > 0:
                    point_in_box_reward = points_in_region_count / len(predict_points)
                
            except Exception as e:
                pass
        
        overall_reward = 0.1*thinking_format_reward + \
            0.1*point_format_reward + \
            0.8*point_in_box_reward
            
        return {
            "overall": overall_reward,
            "thinking_format": thinking_format_reward,
            "point_format": point_format_reward,
            "handal_point_in_mask": point_in_box_reward,
        }

    elif type_value == "3d_position":
        # using mm distance
        point_format_reward = embodied_r1_3d_point_format_reward(predict_str)
        point_position_reward = 0.0

        if point_format_reward:
            try:
                json_data = json.loads(actual_ground_truth)
                object_info = json_data.get("object", [])
                direction = json_data.get("direction", [])
                
                assert isinstance(object_info, list)
                assert type(direction)==str
                for point in object_info:
                    assert isinstance(point, list) and len(point) == 3

                # parse predicted points
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_points = json.loads(point_content)

                # parse predicted depth
                depth_pattern = r'<depth>(.*?)</depth>'
                depth_match = re.search(depth_pattern, predict_str, re.DOTALL)
                depth_content = depth_match.group(1).strip()
                predict_depth = json.loads(depth_content)

                # verify format of predicted points
                assert isinstance(predict_points, list)
                assert isinstance(predict_depth, list)
                assert len(predict_points)==len(predict_depth)

                # build complete x y depth
                point_position_reward = 0.0
                for i in range(len(predict_points)):
                    predict_answer=pixel_to_world(predict_points[i][0], predict_points[i][1], float(predict_depth[i]/1000))
                    point_position_reward += calculate_xy_depth_reward(object_info, direction, predict_answer)
                point_position_reward/=len(predict_points)
            except Exception as e:
                print(f"Error parsing content or calculating reward: {e}, predict_str: {predict_str}")
                pass
        
        overall_reward = 0.1*thinking_format_reward + \
            0.1*point_format_reward + \
            0.8*point_position_reward
                
        return {
            "overall": overall_reward,
            "thinking_format": thinking_format_reward,
            "point_format": point_format_reward,
            "open6dor_3d_position_reward": point_position_reward,
        }
    else:
        raise ValueError(f"Unknown type: {type_value}")