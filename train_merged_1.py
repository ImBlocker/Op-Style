import json
import os
import random
import data_tool1
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from modules import Xfeatmodel1, CPCmodel_PPOGC_NC, hum, CPCmodel_PPOGC, NAXfeatmodel

import math


'''class ChunkedDataset(IterableDataset):
    def __init__(self, folder_path, target_folder, shuffle=True):
        self.folder_path = folder_path
        self.target_folder = target_folder
        self.shuffle = shuffle
        self.json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if self.shuffle:
            random.shuffle(self.json_files)

    def process_file(self, json_file):
        """处理单个文件，返回生成器"""
        file_path = os.path.join(self.folder_path, json_file)
        data = data_tool.load_and_fix_json(file_path)
        if data is None:
            return []
        processed_data = data_tool.process_data(data)
        converted_data = convert_state_to_tensor(processed_data)
        if self.shuffle:
            random.shuffle(converted_data)
        for item in converted_data:
            yield item["state_tensor"]

    def __iter__(self):
        """迭代器：逐个文件加载数据"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程加载
            files = self.json_files
        else:  # 多进程分块
            per_worker = len(self.json_files) // worker_info.num_workers
            worker_id = worker_info.id
            files = self.json_files[worker_id * per_worker : (worker_id + 1) * per_worker]

        for json_file in files:
            yield from self.process_file(json_file)
'''

class SequentialChunkedDataset(IterableDataset):
    def __init__(self, folder_path, shuffle_files=False):
        self.folder_path = folder_path
        self.shuffle_files = shuffle_files
        self.json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if self.shuffle_files:
            random.shuffle(self.json_files)  # 文件间可打乱顺序，但文件内时序固定

    def process_file(self, json_file):
        """处理单个文件，返回该文件的所有数据（保持时序）"""
        file_path = os.path.join(self.folder_path, json_file)
        data = data_tool1.load_and_fix_json(file_path)
        if data is None:
            return []
        # processed_data = data_tool.process_data(data)
        converted_data = convert_state_to_tensor(data)
        
        # 调用 hum 函数生成特征
        hum_features = hum.count_occurrences(data)

        # 调试：检查张量形状
        # for state_item, hum_feature in zip(converted_data, hum_features):
        #     print("state_tensor shape:", state_item["state_tensor"].shape)
            # print("hum_feature shape:", hum_feature.shape)

        # 合并 state_tensor 和 hum_feature
        combined_data = []
        for state_item, hum_feature in zip(converted_data, hum_features):
            # print("111", type(hum_feature["p2"]))
            # print("222", hum_feature["feat"])
            state_item["hum_feature"] = hum_feature["feat"]
            combined_data.append(state_item)

        return combined_data


    def __iter__(self):
            """迭代器：逐个文件加载数据"""
            for json_file in self.json_files:
                file_data = self.process_file(json_file)
                for item in file_data:
                    yield item  # 返回包含 state_tensor 和 hum_feature 的字典

def convert_state_to_tensor(data):
    converted_data = []
    for step_info in data:
        state = step_info["state"]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.permute(1, 0).view(12, 16, 16)
        converted_data.append({"step": step_info["step"], "state_tensor": state_tensor})
    return converted_data

'''def train_chunked_model(ckpt_save_path, batch_size=10, n_steps=160000, lr=3e-4, gamma_steplr=0.5, device_num='0'):
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    encoder = Xfeatmodel.XFeatModel().to(device)
    cdc_model = CPCmodel.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=encoder).to(device)
    
    # 优化器和学习率调度器
    optimizer = Adam(list(encoder.parameters()) + list(cdc_model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=n_steps // 10, gamma=gamma_steplr)
    
    # 创建分块数据集
    dataset = ChunkedDataset(folder_path="aa", target_folder="bb", shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)  # 多进程加速
    
    # 训练循环
    for step in range(n_steps):
        for batch in dataloader:
            batch = batch.to(device)
            # 前向传播
            features = encoder(batch)
            accuracy, nce_loss, hidden = cdc_model(features, cdc_model.init_hidden(batch_size))
            # 反向传播
            optimizer.zero_grad()
            nce_loss.backward()
            optimizer.step()
        # 更新学习率
        scheduler.step()
        # 保存检查点
        if step % 500 == 0:
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'cdc_model_state_dict': cdc_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': step,
            }, os.path.join(ckpt_save_path, f"checkpoint_{step}.pt"))
    
    print("训练完成，模型已保存。")
'''

def custom_collate(batch):
    state_tensors = [item["state_tensor"] for item in batch]
    hum_features = [item["hum_feature"] for item in batch]
    return {
        "state_tensor": torch.stack(state_tensors),
        "hum_feature": torch.stack(hum_features)
    }

def train_sequential_model(ckpt_save_path, batch_size=100, n_epochs=10, lr=3e-4, gamma_steplr=0.5, device_num='0', checkpoint_path=None):
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    encoder = Xfeatmodel1.XFeatModel().to(device)
    cdc_model = CPCmodel_PPOGC.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=encoder).to(device)
    
    # 优化器和学习率调度器
    optimizer = Adam(list(encoder.parameters()) , lr=lr)
    scheduler = StepLR(optimizer, step_size=n_epochs // 10, gamma=gamma_steplr)

    # 如果提供了检查点路径，则加载模型状态
    start_epoch = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        cdc_model.load_state_dict(checkpoint['cdc_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {start_epoch}.")

    # 创建数据集
    dataset = SequentialChunkedDataset(folder_path="/mnt/671cbd8b-55cf-4eb4-af6d-a4ab48e8c9d2/JL/JL/PPOGC", shuffle_files=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collate,  # 使用自定义 collate 函数
        drop_last = True
    )

    os.makedirs(ckpt_save_path, exist_ok=True)
    # 创建日志文件
    log_file_path = os.path.join(ckpt_save_path, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch\tLoss\tAccuracy\n")  # 写入表头
    
    # 训练循环
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        batch_count = 0

        hidden = None
        for batch in dataloader:
            # 分离 state_tensor 和 hum_feature
            state_tensor = batch["state_tensor"].to(device)  # 形状: (batch_size, 12, 16, 16)
            hum_features = batch["hum_feature"].to(device)   # 形状: (batch_size, 4)
            
            # 初始化隐藏状态
            if hidden is None:
                hidden = cdc_model.init_hidden(batch_size)
            
            # 前向传播（传递 hum_features）
            accuracy, nce_loss, hidden = cdc_model(state_tensor, hum_features, hidden.detach())

            # 累计指标
            total_loss += nce_loss.item()
            total_accuracy += accuracy
            batch_count += 1

            # 反向传播
            optimizer.zero_grad()
            nce_loss.backward()
            optimizer.step()

        avg_loss = total_loss / batch_count
        avg_acc = total_accuracy / batch_count
        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")

        # 写入日志文件
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{epoch + 1}\t{avg_loss:.4f}\t{avg_acc:.2%}\n")

        # 更新学习率并保存模型
        scheduler.step()
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'cdc_model_state_dict': cdc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }, os.path.join(ckpt_save_path, f"checkpoint_epoch_{epoch}.pt"))

    
    print("训练完成，模型已保存。")
    
# 主流程
if __name__ == "__main__":
    train_sequential_model(
        ckpt_save_path=os.path.join('xcmd/PPOGC'),
        batch_size=50,
        n_epochs=100,
        # lr=3e-4,
        lr=3e-4 * math.sqrt(5),
        gamma_steplr=0.5,
        device_num='0',
        checkpoint_path=''
    )