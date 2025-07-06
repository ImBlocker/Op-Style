import torch
from torch.utils.data import Dataset
import json

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        # 加载 JSON 数据
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        # 数据集的长度是总步数减一（因为需要相邻两步）
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 获取当前步和下一步的数据
        current_step = self.data[idx]
        next_step = self.data[idx + 1]

        # 提取相邻两步的 tenser 数据
        tenser1 = torch.tensor(current_step['state_tensor'], dtype=torch.float32)  # 当前步的 tenser
        tenser2 = torch.tensor(next_step['state_tensor'], dtype=torch.float32)     # 下一步的 tenser

        # 返回相邻两步的 tenser 数据
        return tenser1, tenser2

# # 示例用法
# if __name__ == '__main__':
#     # 假设 JSON 文件路径
#     data_path = 'path/to/your/dataset.json'

#     # 初始化数据集
#     dataset = CustomDataset(data_path)

#     # 打印数据集长度
#     print(f"Dataset length: {len(dataset)}")

#     # 获取第一个样本
#     tenser1, tenser2 = dataset[0]
#     print(f"Tenser 1: {tenser1}")
#     print(f"Tenser 2: {tenser2}")