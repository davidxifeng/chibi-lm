import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码：为每个位置生成固定的编码向量

    为什么需要？因为注意力机制本身不感知顺序。
    Q·K 的计算对token的位置不敏感，所以需要显式加入位置信息。
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # [max_len, 1]

        # 使用sin和cos函数生成位置编码
        # 公式: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        #       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos

        # 注册为buffer（不是参数，不需要梯度）
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + position_encoding: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 直接加到输入上
        return x + self.pe[:seq_len, :].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    核心思想：用多个"头"并行地关注不同的信息
    每个头有独立的Q、K、V变换，最后拼接起来
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q、K、V的线性变换（所有头共用一个大矩阵，然后拆分）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [seq_len, seq_len] - Causal mask，确保只能看到前面的token
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性变换得到Q、K、V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 拆分成多个头
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch, n_heads, seq_len, seq_len]

        # 4. 应用causal mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. Softmax得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)

        # 6. 加权求和: attn_weights @ V
        attn_output = torch.matmul(attn_weights, V)
        # [batch, n_heads, seq_len, d_k]

        # 7. 合并多个头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 8. 输出投影
        output = self.W_o(attn_output)

        return output


class FeedForward(nn.Module):
    """
    前馈网络：两层全连接，中间加激活函数

    FFN(x) = ReLU(xW1 + b1)W2 + b2
    通常中间层维度是d_model的4倍
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """
    单个Transformer层

    结构：
    1. LayerNorm -> Multi-Head Attention -> 残差连接
    2. LayerNorm -> Feed Forward -> 残差连接
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 注意力子层
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)  # 残差连接

        # 前馈子层
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)  # 残差连接

        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer (类似GPT)

    结构：
    Token Embedding + Position Encoding
      ↓
    N × TransformerBlock
      ↓
    Linear (投影到vocab_size)
    """

    def __init__(
        self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # N个Transformer层
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # 输出层：投影到词汇表大小
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_causal_mask(self, seq_len):
        """
        生成因果掩码（Causal Mask）

        确保位置i只能看到位置<=i的token（不能看到未来）

        返回: [seq_len, seq_len]
        例如seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] - token索引
        Returns:
            logits: [batch, seq_len, vocab_size] - 每个位置对下一个token的预测
        """
        batch_size, seq_len = x.shape

        # 1. Token embedding
        x = self.embedding(x) * math.sqrt(self.d_model)  # 缩放（GPT论文中的技巧）

        # 2. 加入位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 3. 生成causal mask
        mask = self.generate_causal_mask(seq_len).to(x.device)

        # 4. 通过所有Transformer层
        for layer in self.layers:
            x = layer(x, mask)

        # 5. 投影到词汇表
        logits = self.fc_out(x)

        return logits


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("Transformer 模型测试")
    print("=" * 70)

    with open("./data/vocab.json", "r", encoding="utf-8") as f:
        import json

        vocab_data = json.load(f)
        word_to_idx = vocab_data["word_to_idx"]
        idx_to_word = vocab_data["idx_to_word"]

    # 模型配置
    config = {
        "vocab_size": 20,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "d_ff": 64,
        "max_len": 16,
        "dropout": 0.1,
    }

    # 创建模型
    model = DecoderOnlyTransformer(**config)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 测试前向传播
    batch_size = 4
    seq_len = 8
    dummy_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    input_text = [
        " ".join([idx_to_word[str(idx.item())] for idx in sequence])
        for sequence in dummy_input
    ]

    print(f"\n测试前向传播:")
    print(f"  输入 shape: {dummy_input.shape}")
    print(f"  输入 : {input_text}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  输出 shape: {output.shape}")
    print(
        f"  输出含义: [batch={batch_size}, seq_len={seq_len}, vocab_size={config['vocab_size']}]"
    )
    print(f"  即：对每个位置，预测下一个token的概率分布")

    i = 0
    for seq in output:
        print(f"input text: {input_text[i]}")
        i = i + 1
        j = 0
        for x in seq:
            probs = F.softmax(x, dim=-1)
            max_idx = torch.argmax(probs)
            next_word = idx_to_word[str(max_idx.item())]
            print(f"next token at {j}: {max_idx}, word: {next_word}")
            j = j + 1

    # 测试causal mask
    print(f"\n因果掩码 (seq_len=5):")
    mask = model.generate_causal_mask(5)
    print(mask)
    print("  解释：1表示可以看到，0表示看不到（未来的token）")

    print("\n✓ 模型测试通过！")
    print("\n下一步: 编写训练循环")
