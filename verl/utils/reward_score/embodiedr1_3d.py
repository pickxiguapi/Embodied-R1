import re
import json
import numpy as np


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

#xydepth判断得分
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

#xydepth得分
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

#假设预测的深度为mm单位
def embodiedr1_3d_compute_score(predict_str: str, ground_truth: str) -> float:
    # 解析ground_truth中的type和实际答案
    type_value, actual_ground_truth = parse_ground_truth(ground_truth)

    # 1. thinking_format_reward
    thinking_format_reward = embodied_r1_thinking_format_reward(predict_str)

    # 预测答案需要包含在<think></think>和<answer></answer>标签中
    # 点的坐标需要使用<point>[[x1,y1],[x2,y2],...,[xn,yn]]</point>格式提供
    # 深度需要使用<depth>[depth1,depth2,...,depthn]</depth>格式提供
    if type_value == "3d_position":
        point_format_reward = robopoint_point_format_reward(predict_str)
        point_distance_reward = 0.0

        if point_format_reward:
            try:
                json_data = json.loads(actual_ground_truth)
                object_info = json_data.get("object", [])
                direction = json_data.get("direction", [])
                
                # 验证ground truth数据格式
                assert isinstance(object_info, list)
                assert type(direction)==str
                for point in object_info:
                    assert isinstance(point, list) and len(point) == 3

                # 解析预测的点
                point_pattern = r'<point>(.*?)</point>'
                point_match = re.search(point_pattern, predict_str, re.DOTALL)
                point_content = point_match.group(1).strip()
                predict_points = json.loads(point_content)

                #解析预测的深度
                depth_pattern = r'<depth>(.*?)</depth>'
                depth_match = re.search(depth_pattern, predict_str, re.DOTALL)
                depth_content = depth_match.group(1).strip()
                predict_depth = json.loads(depth_content)

                # 验证预测点的格式
                assert isinstance(predict_points, list)
                assert isinstance(predict_depth, list)
                assert len(predict_points)==len(predict_depth)

                #构建完整的 x y depth
                point_distance_reward=0
                for i in range(len(predict_points)):
                    predict_answer=pixel_to_world(predict_points[i][0], predict_points[i][1], float(predict_depth[i]/1000))
                    point_distance_reward += calculate_xy_depth_reward(object_info, direction, predict_answer)
                point_distance_reward/=len(predict_points)
            except Exception as e:
                print(f"Error parsing content or calculating reward: {e}, predict_str: {predict_str}")
                pass

        return {
            "overall": 0.1*thinking_format_reward + 0.1*point_format_reward + 0.8*point_distance_reward,
            "thinking_format": thinking_format_reward,
            "point_format": point_format_reward,
            "3d_position_reward": point_distance_reward,
        }

    else:
        raise ValueError(f"Unknown type: {type_value}")
    
    

    
