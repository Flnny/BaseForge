import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#source: https://blog.csdn.net/m0_55846238/article/details/145726560

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

        pe = torch.zeros(max_len, d_model)  # 创建一个(max_len, d_model)的全零矩阵, 用于保存位置编码值
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 创建一个(max_len, 1)的矩阵, 用于保存位置编码的索引值

        # 创建一个(d_model/2,)的矩阵, 用于储存每个维度的频率因子(每两列的频率因子是相同的, 因此一共有d_model/2个频率因子)(每两列的频率因子是相同的,分别用sin和cos表示)
        # torch.arange(0, d_model, 2).float()相当于生成位置编码公式中的索引i
        # 使用log和exp分开计算能够确保在数值范围内进行线性缩放, 从而避免浮点数溢出或精度丢失
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 计算位置编码
        # 对于维度的偶数列
        pe[:, 0::2] = torch.sin(position * div_term)  # 由广播机制：(max_len, 1)*(d_model/2,)->(max_len, d_model/2)
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


# # 示例用法
d_model = 512  # 例如，模型的维度


# pe = PositionalEncoding(d_model)
#
# # 创建一个随机张量，形状为 (batch_size, seq_len, d_model)
# x = torch.randn(32, 50, d_model)
#
# # 添加位置编码
# x = pe(x)
# print(x.shape)  # 应该输出 torch.Size([32, 50, 512])
#
# # 获取位置编码矩阵
# pe_matrix = pe.pe[0]
#
# # 绘制位置编码矩阵
# plt.figure(figsize=(12, 6))
# plt.imshow(pe_matrix.detach().numpy(), aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('Positional Encoding Matrix')
# plt.xlabel('Embedding Dimension')
# plt.ylabel('Position')
# plt.show()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"  # 确保num_heads能整除d_model

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.h = num_heads

        # 这里定义的 linear 参数是 (d_model, d_model)
        self.q_linear = nn.Linear(d_model, d_model)  # W_Q
        self.k_linear = nn.Linear(d_model, d_model)  # W_K
        self.v_linear = nn.Linear(d_model, d_model)  # W_V
        self.o_linear = nn.Linear(d_model, d_model)  # W_O
        self.dropout = nn.Dropout(dropout)

    #query是我们要查的内容，比如翻译任务中的中文训练集
    def forward(self, q, k, v, mask=None):
        """
             input:
                 q, k, v: (batch_size, seq_len, d_model)
                     对于自注意力, 如果输入序列为 x, 那么 q=x, k=x, v=x
                     对于交叉注意力, 如果序列 x_1 对序列 x_2 做 query, 则 q=x_1, k=x_2, v=x_2
                 mask: (batch_size, 1, 1, seq_len)或(batch_size, 1, seq_len, seq_len)
                     mask有多种形式, 可以使用0、1来mask, 也可以使用True、False来mask, 根据具体代码执行mask
             output:
                 seq: (batch_size, seq_len, d_model)
                 attention: (batch_size, h, len_q, len_k) 每个头均有一个注意力权重矩阵
                     对于自注意力, len_q = len_k = len_v = seq_len
                     对于交叉注意力, len_q = tgt_seq_len , len_k = len_v = src_seq_len
             """
        batch_size = q.size(0)
        # 将原始序列变换为QKV矩阵
        # 以 q 的变换为例。序列 q=x 经过 q_linear 变换后，形状仍然为(batch_size, seq_len, d_model)
        # 使用.view方法用于改变张量形状。这里变换成了(batch_size, seq_len, num_heads, d_k)，即把 d_model 拆成了 num_heads*d_k
        # 使用.transpose方法，将形状进一步变为(batch_size, num_heads, seq_len, d_k)
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1,
                                                                              2)  # (batch_size, seq_len, d_model)->(batch_size, num_heads, seq_len, d_k)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 每个头并行计算相似度得分，相似度矩阵形状为(batch_size, num_heads, len_q, len_k)
        # 即每个头都形成了(len_q, len_k)的 scores，scores 的第一行，意思是第一个位置的 q 对所有位置的 k 的得分，因此后续的 softmax 是按 scores 的行来做的
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 这里我们假设mask中为0的地方是需要遮蔽的地方
            scores = scores.masked_fill(mask == 0, -1e9)  # 通过把掩码的位置设置为一个较大的负数，让掩码位置的softmax趋近于零

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)  # 得到所有batch的每个头的相似度矩阵

        # 相似度矩阵与v相乘得到输出
        output = torch.matmul(attention, v)  # (batch_size, num_heads, seq_len, d_k)

        # 首先将output变为(batch_size, seq_len, num_heads, d_k)
        # .contiguous用于确保张量在内存中是连续的
        # 将张量形状变为(batch_size, seq_len, d_model)，相当于把所有头的结果拼接了起来，即 d_k*num_heads 拼成了 d_model
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_linear(output)  # 使用w_o进行线性变换

        # 最终传出输出和每个头的attention，attention根据需要可用于后续的可视化
        return output, attention


# 示例用法
# mha = MultiHeadAttention(d_model=512, num_heads=8)
#
# x = torch.randn(32, 10, d_model)  # 序列矩阵
#
# output, attention = mha(x, x, x)
#
# print(output.shape)
# print(attention.shape)
#
# # 可视化第0个batch的第0个头的attention
# head_attention = attention[0, 0].detach().numpy()  # 提取第 0 个 batch 的第 0 个头
#
# # 绘制热力图
# plt.imshow(head_attention, cmap='viridis')
# plt.colorbar()
# plt.title('Attention Weights')
# plt.xlabel('Key Position')
# plt.ylabel('Query Position')
# plt.show()
#FeedForward 模块是一个简单的前馈神经网络，通常紧跟在多头注意力机制之后。它由两个线性层和一个激活函数组成，中间包含一个丢弃层（Dropout）以防止过拟合。
class FeedForward(nn.Module):
    # d_ff 默认设置为 2048，更多的中间层节点数可以增加网络的容量，使其能够学习更复杂的函数映射。
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
                input: (batch_size, seq_len, d_model)
                output: (batch_size, seq_len, d_model)
                """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


#NormLayer 模块是一个层归一化层，用于对输入进行归一化处理。层归一化通过对每个样本的所有特征进行归一化，使得每个样本的特征具有相同的均值和标准差，
# 对于一个(n, d_model)的序列，层归一化在每个位置对所有的d_model进行归一化。
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    # LayerNorm是对d_model而言的
    #(batch_size, seq_len, d_model)
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # (batch_size, seq_len, 1)
        std = x.std(-1, keepdim=True)  # (batch_size, seq_len, 1)

        return self.a * (x - mean) / (std + self.eps) + self.b


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
                每个EncoderLayer包括两个子层: 多头注意力层和前馈神经网络层。每个子层都使用了残差连接和层归一化。
                """
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        def forward(self, x, src_mask=None):
            """
            原文中使用: LayerNorm(x + SubLayer(x))
            也有部分实现使用: x + SubLayer(LayerNorm(x))
            这里我们使用原文的实现
            input: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, d_model)
            """

            output, _ = self.attn(x, x, x, mask=src_mask)  # 多头自注意力子层
            x = self.norm_1(x + self.dropout_1(output))  # 层归一化
            x = self.ff(x)  # 前馈神经网络子层
            x = self.norm_2(x + self.dropout_2(x))  # 层归一化
            return x


# 编码器
class Encoder(nn.Module):
    """
    编码器由多个编码器层堆叠而成。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        在原始论文的图 1 和描述中, 作者提到每个子层(Multi-Head Attention 和 Feed-Forward Network)之后会进行 Layer Normalization。
        但是，论文并没有明确提到在整个编码器或解码器之后进行额外的 Layer Normalization。
        许多后续的实现，通常会在编码器和解码器的堆叠之后再进行一次 Layer Normalization。
        """
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        每个DecoderLayer包括三个子层: 自注意力层、编码器-解码器注意力层和前馈神经网络层。每个子层都使用了残差连接和层归一化。
        """
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, memory_mask=None, tgt_mask=None):
        """
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        output_1, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm_1(x + self.dropout_1(output_1))  # 第一个子层：多头自注意力层

        output_2, _ = self.enc_dec_attn(x, enc_output, enc_output, mask=memory_mask)  # k, v来自编码器输出
        x = self.norm_2(x + self.dropout_2(output_2))  # 第二个子层：编码器-解码器注意力层

        x = self.norm_3(x + self.dropout_3(self.ff(x)))  # 第三个子层：前馈神经网络层
        return x


# 解码器
class Decoder(nn.Module):
    """
    解码器由多个解码器层堆叠而成。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, memory_mask=None, tgt_mask=None):
        """
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, enc_output, memory_mask, tgt_mask)
        return self.norm(x)

#Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1,
                 max_len=500):
        super().__init__()
        # src_vocab_size和tgt_vocab_size分别是源序列和目标序列的词典大小
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
    """
    src和tgt为token_id
    src: (batch_size, src_seq_len)
    tgt: (batch_size, tgt_seq_len)
    在 Transformer 模型中, 输入序列通常已经经过填充(padding)处理。
    填充是为了使所有输入序列的长度一致，从而可以将它们放入一个批次中进行处理。
    """
    src = self.dropout(self.positional_encoding(self.src_embedding(src)))
    tgt = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))

    enc_output = self.encoder(src, src_mask)
    dec_output = self.decoder(tgt, enc_output, memory_mask, tgt_mask)

    # 在训练过程中，logits 通常会通过 CrossEntropyLoss 来计算损失，而 CrossEntropyLoss 会在内部应用 softmax
    # 因此这里可以不用softmax，在推理阶段，可以在output后手动加入softmax
    output = self.fc_out(dec_output)
    return output


# 填充掩码
def make_padding_mask(seq, pad_id, return_int=True, true_to_mask=False):
    """
    构造padding mask, 参数设置根据不同的Transformer实现来确定
    Args:
        seq: 需要构造mask的序列(batch, seq_len), 该序列使还未进行Embedding, 里面放的是token_id
        pad_id: 用于填充的特殊字符<PAD>所对应的token_id, 根据不同代码设置
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        mask: (batch, seq_len), 不同的Transformer实现需输入的形状也不同, 根据需要进行后续更改
    """
    mask = (seq == pad_id)  # (batch, seq_len), 在<PAD>的位置上生成True, 真实序列的位置为False

    if true_to_mask is False:
        mask = ~mask

    if return_int:
        mask = mask.int()

    return mask


# 因果掩码
def make_sequence_mask(seq, return_int=True, true_to_mask=False):
    """
    构造sequence mask, 参数设置根据不同的Transformer实现来确定
    Args:
        seq: 需要构造mask的序列(batch, seq_len), 该序列使还未进行Embedding, 里面放的是token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        mask: (seq_len, seq_len), 不同的Transformer实现需输入的形状也不同, 根据需要进行后续更改
    """
    _, seq_len = seq.shape
    mask = torch.tril(torch.ones(seq_len, seq_len))  # (seq_len, seq_len), 下三角为1, 上三角为0
    mask = 1 - mask
    mask = mask.bool()

    if true_to_mask is False:
        mask = ~mask

    if return_int:
        mask = mask.int()

    return mask

# 进一步分别构造src_mask、memory_mask、tgt_mask
def make_src_mask(src, pad_id, return_int=True, true_to_mask=False):
    """构造src_mask

    Args:
        src: 源序列(batch_size, src_len)
        pad_id: 补全符号的token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        src_mask: (batch_size, 1, 1, src_len)
    """
    padding_mask = make_padding_mask(src, pad_id, return_int=return_int, true_to_mask=true_to_mask)
    padding_mask = padding_mask.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(2)
    return padding_mask

def make_memory_mask(src, pad_id, return_int=True, true_to_mask=False):
    """构造memory_mask

    Args:
        src: 源序列(batch_size, src_len)
        pad_id: 补全符号的token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        memory_mask: (batch_size, 1, 1, src_len)
    """
    padding_mask = make_padding_mask(src, pad_id, return_int=return_int, true_to_mask=true_to_mask)
    padding_mask = padding_mask.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(2)
    return padding_mask

def make_tgt_mask(tgt, pad_id, return_int=True, true_to_mask=False):
    """构造tgt_mask

    Args:
        tgt: 目标序列(batch_size, tgt_len)
        pad_id: 补全符号的token_id
        return_int: 是否返回int形式的mask, 默认为True
        true_to_mask: 默认为False, 对于bool mask: True代表在True的位置遮蔽, False代表在False的位置遮蔽。对于int mask: True代表在1的位置遮蔽, False代表在0的位置遮蔽

    Returns:
        tgt_mask: (batch_size, 1, tgt_len, tgt_len)
    """
    padding_mask = make_padding_mask(tgt, pad_id, return_int=return_int, true_to_mask=true_to_mask)  # (batch_size, tgt_len)
    padding_mask = padding_mask.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
    padding_mask = padding_mask.repeat(1, 1, tgt.size(1), 1)  # (batch_size, 1, tgt_len, tgt_len)

    sequence_mask = make_sequence_mask(tgt, return_int=True, true_to_mask=False)  # (tgt_len, tgt_len)
    sequence_mask = sequence_mask.unsqueeze(0)
    sequence_mask = sequence_mask.unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
    sequence_mask = sequence_mask.repeat(tgt.size(0), 1, 1, 1)  # (batch_size, 1, tgt_len, tgt_len)

    # 合并两个mask
    if true_to_mask is False:  # 根据不同类型的mask, 使用"与"或"或"的方式进行合并
        mask = padding_mask & sequence_mask
    else:
        mask = padding_mask | sequence_mask
    return mask


# seq = torch.tensor(
#     [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])  # (batch_size, src_seq_len=12)
#
# # 比较四种不同的padding_mask
# print(make_padding_mask(seq=seq, pad_id=0, return_int=True, true_to_mask=False))
# print(make_padding_mask(seq=seq, pad_id=0, return_int=True, true_to_mask=True))
# print(make_padding_mask(seq=seq, pad_id=0, return_int=False, true_to_mask=False))
# print(make_padding_mask(seq=seq, pad_id=0, return_int=False, true_to_mask=True))
#
# # 展示sequence_mask
# print(make_sequence_mask(seq=seq, return_int=True, true_to_mask=False))
#
#
# # 示例
# src = torch.tensor([[1,1,1,1,1,1,1,1,1,0,0,0],
#                     [1,1,1,1,1,1,1,0,0,0,0,0]])
# tgt = torch.tensor([[1,1,1,1,1,1,1,0,0,0,0],
#                     [1,1,1,1,1,0,0,0,0,0,0]])
#
# src_mask = make_src_mask(src, pad_id=0)
# print('src_mask:')
# print(src_mask)
# print(src_mask.shape)
#
# memory_mask = make_memory_mask(src, pad_id=0)  # memory_mask和src_mask这里是一样的, 但在transformer内部会广播成不同的维度
# print('memory_mask:')
# print(memory_mask)
# print(memory_mask.shape)
#
# tgt_mask = make_tgt_mask(tgt, pad_id=0)
# print('tgt_mask:')
# print(tgt_mask)
# print(tgt_mask.shape)
