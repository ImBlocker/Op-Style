import torch
import torch.nn as nn
import torch.nn.functional as F

class CDCK5(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, encoder):
        super(CDCK5, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep

        # 使用外部传入的编码器
        self.encoder = encoder

        # GRU 时序网络
        self.gru = nn.GRU(68, 40, num_layers=2, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(40, 68) for i in range(timestep)])
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        # 初始化 GRU
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size):
        """返回与模型参数相同设备的隐藏状态"""
        device = next(self.parameters()).device  # 自动获取模型所在的设备
        return torch.zeros(2 * 1, batch_size, 40, device=device)  # ✅ 添加 device=devi

    def forward(self, x, hum_features, hidden):
        """
        前向传播函数。
        输入：
            - x: 输入数据，形状为 (batch_size, 1, seq_len)
            - hidden: GRU 的隐藏状态，形状为 (2, batch_size, 40)
        输出：
            - accuracy: 对比预测任务的准确率
            - nce: 噪声对比估计（NCE）损失
            - hidden: 更新后的 GRU 隐藏状态
        """
        batch = x.size()[0]

        # 通过 XFeatModel 编码器提取特征
        z = self.encoder(x)  # 输出形状: (batch_size, 64, H/8, W/8)
        
        # 将空间维度展平为一维序列
        z = z.view(batch, 64, -1)  # 形状: (batch_size, 64, H/8 * W/8)
        z = z.transpose(1, 2)  # 形状: (batch_size, H/8 * W/8, 64)

        # 将 hum_data 扩展为与 z 的时间步长相同的形状
        hum_data = hum_features.unsqueeze(1).expand(-1, z.size(1), -1)  # 形状: (batch_size, H/8 * W/8, hum_feature_size)

        # 将 hum_data 与 z 进行拼接
        z = torch.cat([z, hum_data], dim=2)  # 形状: (batch_size, H/8 * W/8, 64 + hum_feature_size)

        # 随机采样时间步
        t_samples = torch.randint(z.size(1) - self.timestep, size=(1,)).long()

        # 计算对比损失
        nce = 0  # 初始化 NCE 损失
        encode_samples = torch.empty((self.timestep, batch, 68)).float()  # 存储未来时间步的特征
        for i in range(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 68)  # 提取未来时间步的特征

        # 前向序列
        forward_seq = z[:, :t_samples + 1, :]  # 形状: (batch_size, t_samples + 1, 64)
        output, hidden = self.gru(forward_seq, hidden)  # GRU 输出形状: (batch_size, t_samples + 1, 40)
        c_t = output[:, t_samples, :].view(batch, 40)  # 当前时间步的隐藏状态

        # 预测未来时间步的特征
        pred = torch.empty((self.timestep, batch, 68)).float()  # 存储预测结果
        for i in range(self.timestep):
            decoder = self.Wk[i]
            pred[i] = decoder(c_t)  # 预测未来时间步的特征

        # 计算对比损失和准确率
        correct = 0  # 初始化正确预测的数量

        for i in range(self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # 计算相似度矩阵
            # 正确逻辑应为比较预测和目标的索引，且需确保张量在同一设备上：
            targets = torch.arange(0, batch, device=total.device)  # 确保设备一致
            preds = torch.argmax(self.softmax(total), dim=1)  # 按行取最大值（假设目标在行）
            correct += torch.sum(preds == targets).item()  # 使用 .item() 转换为标量
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # 计算 NCE 损失

        nce /= -1. * batch * self.timestep  # 平均 NCE 损失
        accuracy = correct / (batch * self.timestep)

        return accuracy, nce, hidden

    def predict(self, x, hum_features, hidden):
        """
        预测函数，用于提取特征。
        输入：
            - x: 输入数据，形状为 (batch_size, 1, seq_len)
            - hidden: GRU 的隐藏状态，形状为 (2, batch_size, 40)
        输出：
            - output: GRU 的输出，形状为 (batch_size, seq_len, 40)
            - hidden: 更新后的 GRU 隐藏状态
        """
        batch = x.size()[0]

        # 通过 XFeatModel 编码器提取特征
        z = self.encoder(x)  # 输出形状: (batch_size, 64, H/8, W/8)

        # 将空间维度展平为一维序列
        z = z.view(batch, 64, -1)  # 形状: (batch_size, 64, H/8 * W/8)
        z = z.transpose(1, 2)  # 形状: (batch_size, H/8 * W/8, 64)

        # 将 hum_data 扩展为与 z 的时间步长相同的形状
        hum_data = hum_features.unsqueeze(1).expand(-1, z.size(1), -1)  # 形状: (batch_size, H/8 * W/8, hum_feature_size)

        # 将 hum_data 与 z 进行拼接
        z = torch.cat([z, hum_data], dim=2)  # 形状: (batch_size, H/8 * W/8, 64 + hum_feature_size)

        # 通过 GRU 时序网络
        output, hidden = self.gru(z, hidden)  # GRU 输出形状: (batch_size, H/8 * W/8, 40)

        return output, hidden