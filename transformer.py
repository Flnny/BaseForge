import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#位置编码
# 位置编码 通过正弦和余弦函数为每个位置生成了 独特且平滑的编码，有效地提供了 顺序信息 和 相对位置信息。
# 直接 增加一个维度来表示位置 会导致更多的 参数和复杂性，并无法有效地表示位置之间的关系。
# 位置编码通过 加法操作 与输入特征相结合，能够同时捕捉语义信息和位置顺序，从而提升模型对序列的理解能力。
# 位置编码的设计是 简洁、通用且高效，并且能够有效处理不同长度的输入序列。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        """
        为序列加入位置编码
        Args:
            d_model: 序列矩阵的embedding的维度
            max_len: 位置编码矩阵的最大序列长度, 这个长度可以比实际序列长度长, 相加时只要截取实际序列的长度即可
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)# 创建一个(max_len, d_model)的全零矩阵, 用于保存位置编码值
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)# 创建一个(max_len, 1)的矩阵, 用于保存位置编码的索引值

        # 创建一个(d_model/2,)的矩阵, 用于储存每个维度的频率因子(每两列的频率因子是相同的, 因此一共有d_model/2个频率因子)(每两列的频率因子是相同的,分别用sin和cos表示)
        # torch.arange(0, d_model, 2).float()相当于生成位置编码公式中的索引i
        # 使用log和exp分开计算能够确保在数值范围内进行线性缩放, 从而避免浮点数溢出或精度丢失
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 计算位置编码
        # 对于维度的偶数列
        pe[:, 0::2] = torch.sin(position * div_term)# 由广播机制：(max_len, 1)*(d_model/2,)->(max_len, d_model/2)
        # 对于维度的奇数列
        pe[:, 1::2] = torch.cos(position * div_term)

        #增加一个batch维度, 使其能够与输入张量相加
        pe = pe.unsqueeze(0)  # (max_len, d_model)->(1, max_len, d_model)
        # 将位置编码矩阵注册为模型的缓冲区, 这样它将不会被认为是模型的参数
        # 缓冲区会随着模型一起保存和加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        # 原文3.4节中提到, 为了使得单词嵌入表示相对大一些, 乘sqrt(d_model), 以确保嵌入向量的值不会被位置编码淹没。
        x = x * math.sqrt(self.d_model)
        # 将位置编码添加到输入张量上
        # 位置编码依据max_len生成, 而输入序列长度的seq_len应小于等于max_len
        # 通常会将输入序列补全或截断到统一长度, 让这个长度等于max_len即可
        x = x + self.pe[:, :x.size(1), :]
        return x

# 示例用法
d_model = 512  # 例如，模型的维度
pe = PositionalEncoding(d_model)

# 创建一个随机张量，形状为 (batch_size, seq_len, d_model)
x = torch.randn(32, 50, d_model)

# 添加位置编码
x = pe(x)
print(x.shape)  # 应该输出 torch.Size([32, 50, 512])

# 获取位置编码矩阵
pe_matrix = pe.pe[0]

# 绘制位置编码矩阵
plt.figure(figsize=(12, 6))
plt.imshow(pe_matrix.detach().numpy(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Positional Encoding Matrix')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.show()

