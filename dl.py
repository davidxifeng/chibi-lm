import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class LanguageModelDataset(Dataset):
    """语言模型数据集"""
    def __init__(self, data, word_to_idx, max_len=16):
        self.data = data
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        self.pad_idx = word_to_idx["<PAD>"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取词序列
        words = self.data[idx]
        
        # 转换为索引
        indices = [self.word_to_idx[w] for w in words]
        
        # 方案A: input[:-1], label[1:]
        input_ids = indices[:-1]
        labels = indices[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch, pad_idx):
    """
    批处理函数：将不同长度的序列pad到相同长度
    
    Args:
        batch: List[Dict] - 一个batch的数据
        pad_idx: int - padding索引
    
    Returns:
        Dict[str, Tensor] - 批处理后的数据
    """
    # 提取input_ids和labels
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # padding到batch内最长的序列
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    # 模拟加载数据
    import json
    
    with open('./data/train_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open('./data/vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        word_to_idx = vocab_data['word_to_idx']
    
    print("=" * 70)
    print("Dataset 测试")
    print("=" * 70)
    
    # 创建Dataset
    dataset = LanguageModelDataset(train_data, word_to_idx, max_len=16)
    print(f"数据集大小: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"\n第1个样本:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  input_ids: {sample['input_ids'].tolist()}")
    print(f"  labels: {sample['labels'].tolist()}")
    
    # 创建DataLoader
    pad_idx = word_to_idx["<PAD>"]
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )
    
    print("\n\n" + "=" * 70)
    print("DataLoader 测试")
    print("=" * 70)
    
    # 测试一个batch
    batch = next(iter(dataloader))
    print(f"Batch大小: {batch['input_ids'].shape[0]}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"\ninput_ids (前2条):\n{batch['input_ids'][:2]}")
    print(f"\nlabels (前2条):\n{batch['labels'][:2]}")
    
    print("\n✓ Dataset和DataLoader测试通过！")
    print("\n下一步: 构建Transformer模型")
