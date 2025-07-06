
import os
import torch
# import delete

def count_occurrences(data):
    result = []
    
    for step_data in data:
        step = step_data["step"]
        state = step_data["state"]
        
        # 初始化计数器
        # p1_counts = {5: 0, 6: 0, 7: 0, 8: 0}  # p1，第四位为2、5、6、7、8的个数
        p2_counts = {5: 0, 6: 0, 7: 0, 8: 0}  # p2，第四位为2、5、6、7、8的个数
        
        for row in state:
            if row[0] in [1, 0]:
                continue

            # if row[2] == 1:  # 第三位为1 and (row[3]-2)== row[10]
            #     if (row[10] + 2) in p1_counts:
            #         p1_counts[(row[10] + 2)] += 1
            
            # if row[2] == 3:  # 第三位为3 and (row[3]-2)== row[10]
            #     if (row[10] + 2) in p2_counts:
            #         p2_counts[(row[10] + 2)] += 1

            if row[2] == 3 and row[3] !=2:  # 第三位为3 and (row[3]-2)== row[10]
                if (row[3]) in p2_counts:
                    p2_counts[(row[3])] += 1
                if (row[10] + 2) in p2_counts:
                    p2_counts[(row[10] + 2)] += 1

            # if row[2] == 3:  # 第三位为3 and (row[3]-2)== row[10]
            #     if (row[3]) == 2:
            #         p2_counts[(row[3])] += row[0]
            #     if (row[10] + 2) == 2:
            #         p2_counts[(row[10] + 2)] += row[0]


        # 转换为特征向量
        feature = [p2_counts.get(5, 0), p2_counts.get(6, 0), p2_counts.get(7, 0), p2_counts.get(8, 0)]
        feat=(torch.tensor(feature, dtype=torch.float32))

        # 将统计结果添加到结果中
        result.append({
            "step": step,
            # "p1": p1_counts,
            "p2": p2_counts,
            "feat": feat
        })

    # print("hhh", result)
        
    return result


# # 指定文件夹路径
# folder_path = "aa"
# target_folder = "b1"
# os.makedirs(target_folder, exist_ok=True)

# # 获取文件夹中的所有文件名
# files = os.listdir(folder_path)

# # 过滤出JSON文件
# json_files = [file for file in files if file.endswith(".json")]

# # 按文件名排序（可选）
# json_files.sort()
# i = 8

# # 顺序读取每个JSON文件
# for json_file in json_files:
#     i = i + 1
#     file_path = os.path.join(folder_path, json_file)

#     # 加载并修复 JSON 数据
#     data = data_tool.load_and_fix_json(file_path)
#     if data is None:
#         print(f"文件 {json_file} 无法修复，跳过处理。")
#         continue

#     # 处理数据
#     processed_data = data_tool.process_data(data)
#     processed_data = count_occurrences(processed_data)

#     # 保存所有步骤的特征到一个文件
#     target_path = os.path.join(target_folder, f"{i}.json")
#     with open(target_path, 'w') as f:
#         json.dump(processed_data, f, indent=4)

#     print(f"文件 {json_file} 处理完成。")


# file_path1 = 'bb/2.json'
# with open(file_path1, 'r') as file:
#     data = json.load(file)

# # dim_reduction_datad = data_tool.process_data(data)

# # processed_data = delete.process_data(dim_reduction_datad)

# processed_data = count_occurrences(data)

# with open('a4.json', 'w') as f:
#     json.dump(processed_data, f, indent=4)

# print("count:",count)
print("ok了z")
# 示例使用
# 假设文件内容已经加载到变量data中
# data = json.loads(file_content)
# result = count_occurrences(data)
# print(result)