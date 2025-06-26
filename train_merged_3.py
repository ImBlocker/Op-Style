import json
import os
import random
import data_tool1
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from modules import Xfeatmodel, hum, CPCmodel_PPOGCPM, CPCmodel_PPOGC, CPCmodel_PPOGDI
import math

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

def custom_collate(batch):
    state_tensors = [item["state_tensor"] for item in batch]
    hum_features = [item["hum_feature"] for item in batch]
    return {
        "state_tensor": torch.stack(state_tensors),
        "hum_feature": torch.stack(hum_features)
    }

def cosine_distance_loss(feat_new, feat_old1, feat_old2):
    """
    计算新特征与两个旧特征的余弦距离损失。
    - 输入形状: feat_new, feat_old1, feat_old2 均为 (batch_size, feature_dim)
    """

    # 计算余弦相似性（范围 [-1, 1]，1表示方向相同）
    cos_sim1 = F.cosine_similarity(feat_new, feat_old1, dim=1)  # 形状: (batch_size,)
    cos_sim2 = F.cosine_similarity(feat_new, feat_old2, dim=1)
    
    # 最大化距离：最小化余弦相似性的平均值（使其趋近于 -1）
    distance_loss = (cos_sim1.mean() + cos_sim2.mean()) / 2.0  # 合并两个旧模型的损失
    return distance_loss

# 修改后的训练函数
def train_sequential_model(ckpt_save_path, old_checkpoint_path1, old_checkpoint_path2, batch_size=100, n_epochs=10, lr=3e-4, gamma_steplr=0.5, device_num='0', checkpoint_path=None, lambda_div=0.5):
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    
    # 加载两个旧模型
    old_encoder1 = Xfeatmodel.XFeatModel().to(device)
    old_cpc_model1 = CPCmodel_PPOGCPM.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=old_encoder1).to(device)
    checkpoint_old1 = torch.load(old_checkpoint_path1, map_location=device)
    old_encoder1.load_state_dict(checkpoint_old1['encoder_state_dict'])
    old_cpc_model1.load_state_dict(checkpoint_old1['cdc_model_state_dict'])


    old_encoder2 = Xfeatmodel.XFeatModel().to(device)
    old_cpc_model2 = CPCmodel_PPOGDI.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=old_encoder2).to(device)
    checkpoint_old2 = torch.load(old_checkpoint_path2, map_location=device)
    old_encoder2.load_state_dict(checkpoint_old2['encoder_state_dict'])
    old_cpc_model2.load_state_dict(checkpoint_old2['cdc_model_state_dict'])

    
    # 冻结旧模型
    for model in [old_encoder1, old_cpc_model1]:    #, old_encoder2, old_cpc_model2]:
        for param in model.parameters():
            param.requires_grad = False

    # 初始化新模型
    encoder = Xfeatmodel.XFeatModel().to(device)
    cdc_model = CPCmodel_PPOGCPM.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=encoder).to(device)
    
    # 优化器
    optimizer = Adam(list(encoder.parameters()), lr=lr)
    # scheduler = StepLR(optimizer, step_size=n_epochs // 10, gamma=gamma_steplr)

    # 如果提供了检查点路径，则加载模型状态
    start_epoch = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        cdc_model.load_state_dict(checkpoint['cdc_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {start_epoch}.")

        # 创建数据集
    dataset = SequentialChunkedDataset(folder_path="/mnt/671cbd8b-55cf-4eb4-af6d-a4ab48e8c9d2/JL/JL/passive", shuffle_files=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collate,  # 使用自定义 collate 函数
        drop_last = True
    )
    
    # 创建日志文件
    log_file_path = os.path.join(ckpt_save_path, "training_log.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # 确保目录存在
    with open(log_file_path, "w") as f:
        f.write("Epoch\tNCE Loss\tCosine Dist Loss\tAccuracy\n")  # 写入表头

    # 训练循环
    for epoch in range(start_epoch, n_epochs):
        hidden, hidden_old1, hidden_old2 = None, None, None
        total_nce_loss = 0.0
        total_div_loss = 0.0
        total_accuracy = 0.0  # 新增：记录准确率
        batch_count = 0
        
        for batch in dataloader:
            state_tensor = batch["state_tensor"].to(device)
            hum_features = batch["hum_feature"].to(device)
            
            # 新模型前向
            if hidden is None:
                hidden = cdc_model.init_hidden(batch_size)
            accuracy, nce_loss, hidden, feat_new = cdc_model(state_tensor, hum_features, hidden.detach())
            
            # 旧模型1前向提取特征
            with torch.no_grad():
                if hidden_old1 is None:
                    hidden_old1 = old_cpc_model1.init_hidden(batch_size)
                _, _, hidden_old1, feat_old1 = old_cpc_model1(state_tensor, hum_features, hidden_old1.detach())
            
            # 旧模型2前向提取特征
            with torch.no_grad():
                if hidden_old2 is None:
                    hidden_old2 = old_cpc_model2.init_hidden(batch_size)
                _, _, hidden_old2, feat_old2 = old_cpc_model2(state_tensor, hum_features, hidden_old2.detach())

      # 计算余弦距离损失
            divergence_loss = cosine_distance_loss(feat_new, feat_old1, feat_old2)
            # divergence_loss = F.cosine_similarity(feat_new, feat_old1, dim=1)
            total_loss = nce_loss + lambda_div * divergence_loss  # 总损失

            # # 累计指标
            # total_loss += nce_loss.item()
            total_accuracy += accuracy
            # batch_count += 1

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


            # 记录损失
            total_nce_loss += nce_loss.item()
            total_div_loss += divergence_loss.item()
            batch_count += 1

        # 计算平均损失和准确率
        avg_nce = total_nce_loss / batch_count
        avg_div = total_div_loss / batch_count
        avg_accuracy = total_accuracy / batch_count  # 新增：计算平均准确率
        
        # 打印日志
        print(f"Epoch {epoch+1} | NCE Loss: {avg_nce:.4f} | Cosine Dist Loss: {avg_div:.4f} | Accuracy: {avg_accuracy:.2%}")
        
        # 将日志写入文件
        with open(log_file_path, "a") as f:
            f.write(f"{epoch+1}\t{avg_nce:.4f}\t{avg_div:.4f}\t{avg_accuracy:.2%}\n")
        
        # 保存模型检查点
        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path, exist_ok=True)
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'cdc_model_state_dict': cdc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, os.path.join(ckpt_save_path, f"checkpoint_epoch_{epoch}.pt"))
    
    print("训练完成，模型已保存。")
# 主流程
if __name__ == "__main__":
    train_sequential_model(
        ckpt_save_path=os.path.join('xcmd/Passive2'),
        old_checkpoint_path1='xcmd/PPOGC/checkpoint_epoch_98.pt',
        old_checkpoint_path2='xcmd/PPOGDI2/checkpoint_epoch_120.pt',
        batch_size=50,
        n_epochs=200,
        # lr=3e-4,zheshibushi
        lr=3e-4 * math.sqrt(5),
        gamma_steplr=0.5,
        device_num='0',
        checkpoint_path=''
    )