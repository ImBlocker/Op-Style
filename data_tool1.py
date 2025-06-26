import numpy as np
import json

import os
import json


# def load_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

def fix_json_format(json_str):
    """
    修复 JSON 字符串中的常见格式错误（如缺少逗号）。
    """
    # 修复 }{ 和 ][ 之间缺少逗号的问题
    fixed_json_str = json_str.replace("}{", "},{").replace("][", "],[")
    return fixed_json_str


def load_and_fix_json(file_path):
    """
    读取 JSON 文件并尝试修复格式错误。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_str = f.read()

    try:
        # 尝试直接解析 JSON
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"检测到 JSON 格式错误: {e}")
        # 尝试修复 JSON 格式
        fixed_json_str = fix_json_format(json_str)
        try:
            data = json.loads(fixed_json_str)
            print("已修复 JSON 文件并成功加载。")
        except json.JSONDecodeError as e:
            print(f"修复失败，无法加载 JSON 文件: {e}")
            return None
    return data


# 处理两个四层嵌套数组，生成新的观测数据
def process_observations(obs1, obs2):
    new_obs = []  # 用于存储处理后的观测数据

    # 遍历每个观测组
    for group1, group2 in zip(obs1, obs2):
        new_group = []  # 用于存储处理后的观测组
        # 遍历每个二维数组

        # 遍历矩阵的每一行
        for row1, row2 in zip(group1, group2):
            new_row = []  # 用于存储处理后的行
            # 比较前12位（与操作）和第13位（或操作）
            for col_index in range(len(row1)):
                if col_index < 12:
                    # 前12位进行与操作
                    new_row.append(row1[col_index] and row2[col_index])
                elif col_index == 12:
                    # 第13位进行或操作
                    new_row.append(row1[col_index] or row2[col_index])
                else:
                    # 其余位正常比较
                    new_row.append(row1[col_index] and row2[col_index])

            new_group.append(new_row)
        new_obs.append(new_group)

    return new_obs


def p0_data(data):
    # input_file = 'game_history_20250225_152757.json'  # 原始文件路径
    # output_file = 'processed_observations.json'  # 输出文件路径
    # data = load_json_file(input_file)

    # 遍历每一步，处理观测数据
    for step_data in data:
        observations = step_data['observations']
        if len(observations) < 2:
            print(f"步骤 {step_data['step']}: observations 中观测组数量不足，无法处理。")
            continue

        obs1 = observations[0]
        obs2 = observations[1]

        # 处理两组观测数据
        new_obs = process_observations(obs1, obs2)

        # 将处理后的观测数据存入新的结构中
        step_data['observations'] = new_obs
        step_data['actions'] = step_data['actions'][1]

    return data


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int32):
        return float(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def flatten_to_1d_with_subarrays(multi_dim_array):
    flattened = []
    for sub_array in multi_dim_array:
        if isinstance(sub_array, list) and any(isinstance(item, list) for item in sub_array):
            # 如果子数组是多维的，展平它
            flattened.extend(flatten_to_1d_with_subarrays(sub_array))
        else:
            # 如果子数组是一维的或最小的数组，保持不变
            flattened.append(sub_array)
    return flattened


def dim_reduction(data):
    processed_data = []
    for item in data:
        step = item['step']
        # # 展平observation中的多维数组为一维数组，但保持最小的数组不变
        # flattened_observation = flatten_to_1d_with_subarrays(item['observation'])

        # # 展平action中的多维数组为一维数组，但保持最小的数组不变
        # flattened_action = flatten_to_1d_with_subarrays(item['action'])

        # flattened_observations = flatten_to_1d_with_subarrays(item['p1']['observations'])
        # flattened_actions1 = flatten_to_1d_with_subarrays(item['p1']['actions'])
        # flattened_actions2 = flatten_to_1d_with_subarrays(item['p2']['actions'])

        flattened_observations = flatten_to_1d_with_subarrays(item['observations'])
        flattened_actions1 = flatten_to_1d_with_subarrays(item['actions'])

        processed_item = {
            'step': step,
            'observations': flattened_observations,
            'action': flattened_actions1,
            'rewards': item['rewards']
        }
        processed_data.append(processed_item)

    return processed_data


def merge_observations_and_actions(data):
    """
    将JSON数据中的p1的observations和p2的action一一对应合并。
    如果长度不一致，用0填充较短的部分。
    合并后的数据命名为`state`，并保留`step`字段。
    """
    merged_data = []

    for step_info in data:
        step = step_info['step']
        observations = step_info['observations']
        actions = step_info['action']

        # 确定每个step中p1和p2的最大长度
        max_length = max(len(observations), len(actions))

        # 初始化合并后的state
        merged_step = []

        for i in range(max_length):
            # 获取p1的observation，如果超出范围则用0填充
            observation = observations[i] if i < len(observations) else [0] * len(observations[0])

            # 获取p2的action，如果超出范围则用0填充
            action = actions[i] if i < len(actions) else [0] * len(actions[0])

            # 如果p1的observation全为0，则将对应的p2的action也设置为全0
            if all(obs == 0 for obs in observation):
                action = [0] * len(action)

            # 合并p1的observation和p2的action
            state = observation + action
            merged_step.append(state)

        # 保留step字段，并将合并后的数据命名为stat,e
        merged_data.append({
            "step": step,
            "state": merged_step
        })

    return merged_data


# def count_state_arrays(data):

#     for step_info in data:
#     for step_info in data:
#         state = step_info.get("state", [])
#         state_count = len(state)  # 统计state数组的个数

#     return state_count

# def merge_ob(data):
#     """
#     处理每个step中的state数据。
#     将state的前27位按照[5, 5, 3, 8, 6]分组，每组中找到值为1的索引作为新的值，
#     最后用这5个新值替换原state的前27位。
#     """
#     group_sizes = [5, 5, 3, 8, 6]  # 分组大小
#     total_size = sum(group_sizes)  # 总共27位

#     result = []

#     for step_info in data:
#         step = step_info["step"]
#         states = step_info["state"]
#         processed_states = []

#         for state in states:
#             new_state = state[total_size:]  # 保留原state的第27位之后的部分
#             start = 0

#             # 处理前27位，按照分组方式
#             for group_size in group_sizes:
#                 group = state[start:start + group_size]
#                 index = group.index(1) if 1 in group else -1  # 找到值为1的索引，如果没有则为0
#                 new_state.insert(0, index+1)  # 将新值插入到新state的前面
#                 start += group_size

#             processed_states.append(new_state)

#         result.append({"step": step, "state": processed_states})

#     return result

def process_state(state):
    # 前27位
    first_27 = state[:29]

    # 分组
    groups = [
        first_27[:5],  # 前5位
        first_27[5:10],  # 接下来的5位
        first_27[10:13],  # 接下来的3位
        first_27[13:21],  # 接下来的8位
        first_27[21:27]  # 最后的6位
    ]

    # 生成新的5个数字
    new_numbers = []
    for group in groups:
        # print("当前组", group)
        # 找到值为1的下标，并加1
        indices = [i + 1 for i, val in enumerate(group) if val == 1]
        # 如果全为0，则用0替代
        if not indices:
            new_numbers.append(0.0)
            # print("下标 0")
            # break
        else:
            # print("下标", int(''.join(map(str, indices))))
            # 将下标合并为一个数字
            num = int(''.join(map(str, indices)))
            new_numbers.append(float(num))
        # print("end")
    # 替换前29位
    new_state = new_numbers + state[29:]
    return new_state


def merge_ob(data):
    # 处理每个step中的state
    for step in data:
        step["state"] = [process_state(state) for state in step["state"]]

    return data


def process_data(data):
    processed_data = p0_data(data)  # 检查格式
    processed_data = convert_to_serializable(processed_data)  # 归一格式
    processed_data = dim_reduction(processed_data)  # 降维
    processed_data = merge_observations_and_actions(processed_data)  # 合并观察动作
    processed_data = merge_ob(processed_data)  # 减小观察位数

    return processed_data

# file_path1 = 'game_history_20250225_152757.json'
# with open(file_path1, 'r') as file:
#     data = json.load(file)

# processed_data = dim_reduction(data)


# with open('a7.json', 'w') as f:
#     json.dump(processed_data, f, indent=4)

# # print("count:",count)
# print("ok了z")


# # 指定文件夹路径
# folder_path = "/mnt/671cbd8b-55cf-4eb4-af6d-a4ab48e8c9d2/JL/PPOGDI-2"
# target_folder = "/home/hello/sth/rts/MicroRTS-Py/experiments/JL/PPOGDI"
# os.makedirs(target_folder, exist_ok=True)
# # print("hh")
# # 获取文件夹中的所有文件名
# files = os.listdir(folder_path)
#
# # 过滤出JSON文件
# json_files = [file for file in files if file.endswith(".json")]
#
# # 按文件名排序（可选）
# json_files.sort()
# i = 0
# print(i)
# # print(json_files)
# # 顺序读取每个JSON文件s
# for json_file in json_files:
#     i = i + 1
#     print(i)
#     file_path = os.path.join(folder_path, json_file)
#     # with open(file_path, "r", encoding="utf-8") as f:
#     #     data = json.load(f)
#
#     data = load_and_fix_json(file_path)
#     if data is None:
#         print(f"文件 {json_file} 无法修复，跳过处理。")
#         continue
#
#     processed_data = process_data(data)
#
#     # tu_data = convert_state_to_tensor(processed_data)
#
#     target_path = os.path.join(target_folder, f"{i}.json")
#
#     # print(tu_data)
#     print("ok了")
#
#     with open(target_path, 'w') as f:
#         json.dump(processed_data, f, indent=4)
#
# print("end")