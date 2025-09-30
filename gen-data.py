import random
from typing import List, Tuple

# ============================================
# 词汇定义
# ============================================

vocab = {
    "animals": ["猫", "狗", "鸟", "鱼"],
    "foods": ["骨头", "虫子", "草"],
    "fruits": ["苹果", "橙子"],
    "actions": ["吃", "跑", "飞", "叫", "跳", "玩耍", "游泳"],
    "categories": ["动物", "食物", "水果"],
    "functional": ["是", "，", "。", "<EOS>"],
    "special": ["<PAD>"]
}

# 扁平化词汇表（用于后续编码）
all_words = (
    vocab["animals"] + vocab["foods"] + vocab["fruits"] + 
    vocab["actions"] + vocab["categories"] + 
    vocab["functional"] + vocab["special"]
)

# 创建词到索引的映射
word_to_idx = {word: idx for idx, word in enumerate(all_words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print("=" * 70)
print("词汇表")
print("=" * 70)
print(f"总词汇量: {len(all_words)}")
for category, words in vocab.items():
    print(f"{category:12s}: {words}")
print(f"\nword_to_idx 示例: {list(word_to_idx.items())[:5]}")

# ============================================
# 语义规则定义
# ============================================

# 定义合理的"X吃Y"组合
eating_rules = {
    "猫": ["鱼", "草"],  # 猫主要吃鱼，偶尔吃草
    "狗": ["骨头", "草"],
    "鸟": ["虫子", "草"],
    "鱼": ["虫子"],  # 鱼吃虫子
}

# 定义合理的"X跑"、"X飞"等无宾语动作
action_rules = {
    "跑": ["猫", "狗"],
    "飞": ["鸟"],
    "叫": ["猫", "狗", "鸟"],  # 三种动物都会叫
    "跳": ["猫", "狗", "鸟"],  # 都能跳
    "玩耍": ["猫", "狗"],
    "游泳": ["狗", "鱼"],  # 狗和鱼会游泳
}

# 类别归属
category_rules = {
    "猫": "动物", "狗": "动物", "鸟": "动物", "鱼": "动物",
    "骨头": "食物", "虫子": "食物", "草": "食物",
    "苹果": "水果", "橙子": "水果",
}

# ============================================
# 句子生成函数
# ============================================

def generate_category_sentence(entity: str, add_eos: bool = True) -> List[str]:
    """生成 'X 是 Y' 句子"""
    category = category_rules[entity]
    sent = [entity, "是", category]
    if add_eos:
        sent.append("<EOS>")
    return sent

def generate_eating_sentence(animal: str, add_eos: bool = True) -> List[str]:
    """生成 'X 吃 Y' 句子"""
    possible_foods = eating_rules.get(animal, [])
    if not possible_foods:
        return None
    food = random.choice(possible_foods)
    sent = [animal, "吃", food]
    if add_eos:
        sent.append("<EOS>")
    return sent

def generate_action_sentence(action: str, add_eos: bool = True) -> List[str]:
    """生成 'X 跑/飞' 句子（无宾语）"""
    possible_subjects = action_rules.get(action, [])
    if not possible_subjects:
        return None
    subject = random.choice(possible_subjects)
    sent = [subject, action]
    if add_eos:
        sent.append("<EOS>")
    return sent

def generate_training_sample() -> Tuple[List[str], str]:
    """生成一条完整的训练样本（单句或多句）"""
    
    sample_type = random.random()
    
    # 40%概率：单句 - "X是Y。<EOS>"
    if sample_type < 0.4:
        entity = random.choice(list(category_rules.keys()))
        sent = generate_category_sentence(entity, add_eos=False)
        full_sentence = sent + ["。", "<EOS>"]
    
    # 40%概率：单句 - "X动作Y。<EOS>" 或 "X动作。<EOS>"
    elif sample_type < 0.8:
        choice = random.random()
        if choice < 0.7:  # 吃的句子
            animal = random.choice(list(eating_rules.keys()))
            sent = generate_eating_sentence(animal, add_eos=False)
        else:  # 跑/飞的句子
            action = random.choice(list(action_rules.keys()))
            sent = generate_action_sentence(action, add_eos=False)
        
        if sent is None:
            # 回退到吃的句子
            animal = random.choice(vocab["animals"])
            sent = generate_eating_sentence(animal, add_eos=False)
        
        full_sentence = sent + ["。", "<EOS>"]
    
    # 20%概率：多句 - "X是Y，X动作Z。<EOS>"（逗号连接，句号结束）
    else:
        entity = random.choice(vocab["animals"])
        sent1 = generate_category_sentence(entity, add_eos=False)
        
        choice = random.random()
        if choice < 0.7:
            sent2 = generate_eating_sentence(entity, add_eos=False)
        else:
            possible_actions = [act for act, subjects in action_rules.items() if entity in subjects]
            if possible_actions:
                action = random.choice(possible_actions)
                sent2 = generate_action_sentence(action, add_eos=False)
            else:
                sent2 = generate_eating_sentence(entity, add_eos=False)
        
        if sent2 is None:
            sent2 = generate_eating_sentence(entity, add_eos=False)
        
        # 用逗号连接，句号结束，最后加<EOS>
        full_sentence = sent1 + ["，"] + sent2 + ["。", "<EOS>"]
    
    # 转为字符串形式（用于去重）
    sentence_str = " ".join(full_sentence)
    
    return full_sentence, sentence_str

# ============================================
# 生成训练数据集
# ============================================

def generate_dataset(n_samples: int = 100) -> List[List[str]]:
    """生成n_samples条不重复的训练样本"""
    dataset = []
    seen = set()
    
    attempts = 0
    max_attempts = n_samples * 10  # 防止无限循环
    
    while len(dataset) < n_samples and attempts < max_attempts:
        sample, sample_str = generate_training_sample()
        if sample_str not in seen:
            dataset.append(sample)
            seen.add(sample_str)
        attempts += 1
    
    if len(dataset) < n_samples:
        print(f"\n⚠️  警告: 只能生成 {len(dataset)} 条不重复样本（目标 {n_samples} 条）")
    
    return dataset

# ============================================
# 生成并显示数据
# ============================================

print("\n\n" + "=" * 70)
print("生成训练数据")
print("=" * 70)

# 生成数据集
train_data = generate_dataset(n_samples=100)

print(f"\n成功生成 {len(train_data)} 条训练样本")
print("\n前20条样本：")
print("-" * 70)
for i, sample in enumerate(train_data[:20], 1):
    print(f"{i:2d}. {' '.join(sample)}")

# ============================================
# 数据统计
# ============================================

print("\n\n" + "=" * 70)
print("数据统计")
print("=" * 70)

# 统计序列长度
seq_lengths = [len(sample) for sample in train_data]
print(f"序列长度范围: {min(seq_lengths)} - {max(seq_lengths)}")
print(f"平均长度: {sum(seq_lengths) / len(seq_lengths):.1f}")

# 统计词频
from collections import Counter
word_freq = Counter()
for sample in train_data:
    word_freq.update(sample)

print(f"\n词频统计（Top 10）:")
for word, freq in word_freq.most_common(10):
    print(f"  {word:6s}: {freq:3d} 次")

# ============================================
# 转换为索引序列
# ============================================

def encode_sentence(sentence: List[str]) -> List[int]:
    """将词序列转换为索引序列"""
    return [word_to_idx[word] for word in sentence]

def pad_sequence(seq: List[int], max_len: int, pad_idx: int) -> List[int]:
    """填充序列到固定长度"""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))

print("\n\n" + "=" * 70)
print("编码示例")
print("=" * 70)

# 示例：编码前3条数据
max_seq_len = 16
pad_idx = word_to_idx["<PAD>"]

for i in range(min(3, len(train_data))):
    sample = train_data[i]
    encoded = encode_sentence(sample)
    padded = pad_sequence(encoded, max_seq_len, pad_idx)
    
    print(f"\n样本 {i+1}:")
    print(f"  原始: {' '.join(sample)}")
    print(f"  编码: {encoded}")
    print(f"  填充: {padded}")

# ============================================
# 保存数据（PyTorch风格）
# ============================================

import json
import os

def save_dataset(data, vocab_maps, save_dir="./data"):
    """保存数据集和词汇表到文件"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存训练数据（JSON格式，人类可读）
    data_path = os.path.join(save_dir, "train_data.json")
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ 训练数据已保存到: {data_path}")
    
    # 保存词汇表
    vocab_path = os.path.join(save_dir, "vocab.json")
    vocab_data = {
        "word_to_idx": vocab_maps["word_to_idx"],
        "idx_to_word": vocab_maps["idx_to_word"],
        "vocab_size": len(vocab_maps["word_to_idx"])
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 词汇表已保存到: {vocab_path}")
    
    return data_path, vocab_path

def load_dataset(save_dir="./data"):
    """从文件加载数据集和词汇表"""
    data_path = os.path.join(save_dir, "train_data.json")
    vocab_path = os.path.join(save_dir, "vocab.json")
    
    # 加载训练数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ 已加载 {len(data)} 条训练样本")
    
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # JSON会把整数key转成字符串，需要转回来
    idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
    
    print(f"✓ 词汇量: {vocab_data['vocab_size']}")
    
    return data, vocab_data["word_to_idx"], idx_to_word

# 保存数据
print("\n\n" + "=" * 70)
print("保存数据集")
print("=" * 70)

vocab_maps = {
    "word_to_idx": word_to_idx,
    "idx_to_word": {str(k): v for k, v in idx_to_word.items()}  # JSON需要字符串key
}

data_path, vocab_path = save_dataset(train_data, vocab_maps)

# 测试加载
print("\n测试加载功能...")
loaded_data, loaded_w2i, loaded_i2w = load_dataset()
assert len(loaded_data) == len(train_data), "数据加载失败！"
print("✓ 数据加载测试通过！")

print("\n\n" + "=" * 70)
print("数据集准备完成！")
print("=" * 70)
print(f"""
总结：
- 训练样本数: {len(train_data)}
- 词汇量: {len(all_words)}
- 最大序列长度: {max(seq_lengths)}
- 建议 max_seq_len: {max(seq_lengths) + 2}

文件位置：
- 训练数据: {data_path}
- 词汇表: {vocab_path}

下一步：
1. 创建 PyTorch Dataset 类
2. 使用 DataLoader 进行批处理
3. 开始训练模型
""")
