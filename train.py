import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime

# 假设这些类已经定义（从之前的代码导入）
from dl import LanguageModelDataset, collate_fn
from model import DecoderOnlyTransformer


def load_data(data_dir="./data"):
    """加载训练数据和词汇表"""
    with open(os.path.join(data_dir, "train_data.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        word_to_idx = vocab_data["word_to_idx"]
        idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}

    return train_data, word_to_idx, idx_to_word


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备上
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        logits = model(input_ids)  # [batch, seq_len, vocab_size]

        # 计算损失
        # 需要reshape: logits变为[batch*seq_len, vocab_size], labels变为[batch*seq_len]
        loss = criterion(
            logits.view(-1, logits.size(-1)),  # [batch*seq_len, vocab_size]
            labels.view(-1),  # [batch*seq_len]
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 统计
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    return avg_loss


def generate_text(
    model, start_tokens, idx_to_word, word_to_idx, max_len=20, device="cpu"
):
    """
    生成文本（简单贪心解码）

    Args:
        model: 训练好的模型
        start_tokens: List[str] - 起始词序列
        idx_to_word: 索引到词的映射
        word_to_idx: 词到索引的映射
        max_len: 最大生成长度
        device: 设备

    Returns:
        List[str] - 生成的完整序列
    """
    model.eval()

    # 转换为索引
    tokens = [word_to_idx[w] for w in start_tokens]

    with torch.no_grad():
        for _ in range(max_len):
            # 当前序列
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

            # 前向传播
            logits = model(input_tensor)  # [1, seq_len, vocab_size]

            # 取最后一个位置的预测
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # 贪心选择概率最高的token
            next_token = torch.argmax(next_token_logits).item()

            # 添加到序列
            tokens.append(next_token)

            # 如果生成了<EOS>，停止
            if idx_to_word[next_token] == "<EOS>":
                break

    # 转换回词
    generated = [idx_to_word[idx] for idx in tokens]
    return generated


def save_checkpoint(model, optimizer, epoch, loss, save_dir="./checkpoints"):
    """保存模型检查点"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, save_path)
    print(f"✓ 检查点已保存: {save_path}")


def main():
    # ============================================
    # 配置
    # ============================================

    config = {
        # 模型参数
        "vocab_size": 24,  # 会根据实际词汇表大小调整
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "d_ff": 64,
        "max_len": 16,
        "dropout": 0.1,
        # 训练参数
        "batch_size": 4,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "save_every": 10,  # 每N个epoch保存一次
        # 其他
        "data_dir": "./data",
        "checkpoint_dir": "./checkpoints",
        "device": "mps" if torch.mps.is_available() else "cpu",
    }

    print("=" * 70)
    print("开始训练")
    print("=" * 70)
    print(f"设备: {config['device']}")
    print(f"配置: {config}")

    # ============================================
    # 加载数据
    # ============================================

    print("\n加载数据...")
    train_data, word_to_idx, idx_to_word = load_data(config["data_dir"])
    print(f"训练样本数: {len(train_data)}")
    print(f"词汇量: {len(word_to_idx)}")

    # 更新vocab_size（以防配置不对）
    config["vocab_size"] = len(word_to_idx)

    # ============================================
    # 创建数据加载器
    # ============================================

    # 这里需要导入之前写的Dataset类
    dataset = LanguageModelDataset(train_data, word_to_idx, config["max_len"])
    pad_idx = word_to_idx["<PAD>"]
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )

    # 为了演示，这里用占位符
    print("创建DataLoader...")
    print(f"Batch大小: {config['batch_size']}")

    # ============================================
    # 创建模型
    # ============================================

    print("\n初始化模型...")
    model = DecoderOnlyTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
        dropout=config["dropout"],
    )
    model = model.to(config["device"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # ============================================
    # 创建优化器和损失函数
    # ============================================

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 交叉熵损失，忽略padding位置
    pad_idx = word_to_idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"优化器: Adam, 学习率={config['learning_rate']}")
    print(f"损失函数: CrossEntropyLoss")

    # ============================================
    # 训练循环
    # ============================================

    print("\n" + "=" * 70)
    print("开始训练循环")
    print("=" * 70)

    for epoch in range(1, config["num_epochs"] + 1):
        #     # 训练一个epoch
        train_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, config["device"]
        )

        print(f"Epoch {epoch}/{config['num_epochs']} - Loss: {train_loss:.4f}")
        #
        #     # 定期保存
        if epoch % config["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, config["checkpoint_dir"]
            )
        #
        #     # 定期生成样本（检查训练效果）
        if epoch % 1 == 0:

            def gen_and_show(start_tokens):
                generated = generate_text(
                    model,
                    start_tokens,
                    idx_to_word,
                    word_to_idx,
                    max_len=15,
                    device=config["device"],
                )
                print(f"  输入: {start_tokens}")
                print(f"  生成: {' '.join(generated)}")
                print()

            print("\n生成样本:")
            gen_and_show(["猫"])
            gen_and_show(["狗"])
            gen_and_show(["鱼"])

    # ============================================
    # 训练完成
    # ============================================

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    # 最终保存
    save_checkpoint(
        model, optimizer, config["num_epochs"], train_loss, config["checkpoint_dir"]
    )

    print(f"\n模型已保存到: {config['checkpoint_dir']}")
    print("\n下一步:")
    # 添加从csv文件加载更大数据集，实现简单的分词功能，创建词汇表
    print("1. 加载checkpoint")
    print("2. 生成文本测试")
    print("3. 分析训练曲线")


if __name__ == "__main__":
    main()
