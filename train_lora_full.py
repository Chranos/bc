#!/usr/bin/env python3
# filepath: /workspace/vlm/lab/BLIP2-Chinese/train_lora_fashion_scene.py

"""
BLIP2 Q-Former LoRA 微调脚本 - 时尚场景分类版本
使用 ITC + 场景分类联合损失
- ITC (Image-Text Contrastive): 图文对比学习
- Scene Classification: 10类场景分类

数据集格式适配：
- 数据路径: /workspace/vlm/lab/output/scene_annotations.json
- 图片路径: /data/fasion/train/image/{file_name}
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import time
import random

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from models.blip2_qformer import Blip2Qformer


# ============================================
# 场景类别定义
# ============================================
SCENE_CATEGORIES = [
    '职场正装',      # 0
    '职场休闲',      # 1
    '运动健身',      # 2
    '户外探险',      # 3
    '居家休闲',      # 4
    '社交聚会',      # 5
    '旅行度假',      # 6
    '运动赛事',      # 7
    '婚礼相关',      # 8
    '特殊功能',      # 9
]

SCENE_TO_ID = {name: idx for idx, name in enumerate(SCENE_CATEGORIES)}
ID_TO_SCENE = {idx: name for idx, name in enumerate(SCENE_CATEGORIES)}
NUM_SCENE_CLASSES = len(SCENE_CATEGORIES)

print(f"📋 场景类别 ({NUM_SCENE_CLASSES} 类):")
for idx, name in enumerate(SCENE_CATEGORIES):
    print(f"  {idx}: {name}")


# ============================================
# 数据集定义（适配时尚数据集）
# ============================================
class FashionSceneDataset(Dataset):
    """
    时尚场景分类数据集
    
    数据格式:
    {
        "file_name": "010207.jpg",
        "scene_category": "居家休闲",
        "text": "描述文本",
        "key_features": "关键特征",
        "suitable_occasion": "适用场合"
    }
    """
    def __init__(self, annotation_file, image_dir, transform=None, 
                 use_key_features=False, max_length=77):
        """
        Args:
            annotation_file: JSON 标注文件路径
            image_dir: 图片目录路径
            transform: 图像变换
            use_key_features: 是否使用 key_features 作为文本（否则使用 text + "，适合" + scene_category）
            max_length: 最大文本长度
        """
        print(f"📂 加载数据集: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform or self._default_transform()
        self.use_key_features = use_key_features
        self.max_length = max_length
        
        # 验证和转换场景标签
        valid_data = []
        invalid_count = 0
        
        for idx, item in enumerate(self.data):
            # 检查必需字段
            required_fields = ['file_name', 'scene_category']
            if not all(field in item for field in required_fields):
                print(f"⚠️ 数据项 {idx} 缺少必需字段，跳过")
                invalid_count += 1
                continue
            
            # 验证场景类别
            scene = item['scene_category']
            if scene not in SCENE_TO_ID:
                print(f"⚠️ 未知场景类别 '{scene}' (文件: {item['file_name']})，跳过")
                invalid_count += 1
                continue
            
            # 验证图片文件是否存在
            image_path = os.path.join(self.image_dir, item['file_name'])
            if not os.path.exists(image_path):
                print(f"⚠️ 图片不存在: {image_path}，跳过")
                invalid_count += 1
                continue
            
            # 转换场景为ID
            item['scene_id'] = SCENE_TO_ID[scene]
            valid_data.append(item)
        
        self.data = valid_data
        
        if invalid_count > 0:
            print(f"⚠️ 跳过 {invalid_count} 个无效数据项")
        
        print(f"📊 有效数据: {len(self.data)} 个样本")
        
        # 统计各场景分布
        scene_counts = {}
        for item in self.data:
            scene_id = item['scene_id']
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        
        print(f"\n📈 场景分布:")
        for scene_id in sorted(scene_counts.keys()):
            count = scene_counts[scene_id]
            ratio = count / len(self.data) * 100
            scene_name = ID_TO_SCENE[scene_id]
            print(f"  {scene_name:8s}: {count:4d} ({ratio:5.1f}%)")
    
    def _default_transform(self):
        """默认的图像变换"""
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image_path = os.path.join(self.image_dir, item['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ 加载图像失败: {image_path}, 错误: {e}")
            image = torch.zeros(3, 224, 224)
        
        # ===== 修改：拼接 text 和 scene_category =====
        if self.use_key_features and 'key_features' in item:
            # 使用 key_features
            text = item['key_features']
        else:
            # 拼接 text + "，适合" + scene_category
            text_content = item.get('text', '')
            scene_category = item.get('scene_category', '')
            
            # 格式：text + "，适合" + scene_category
            if text_content and scene_category:
                text = f"{text_content}，适合{scene_category}"
            elif text_content:
                text = text_content
            else:
                text = scene_category
        
        return {
            'image': image,
            'text': text,
            'scene_label': item['scene_id'],
            'file_name': item['file_name'],  # 用于调试
        }

def split_dataset(annotation_file, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    划分训练集、验证集、测试集
    
    Args:
        annotation_file: 原始标注文件
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_file, val_file, test_file: 分割后的文件路径
    """
    print(f"\n📊 划分数据集 (train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%})...")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按场景分层划分
    scene_data = {}
    for item in data:
        scene = item.get('scene_category', 'unknown')
        if scene not in scene_data:
            scene_data[scene] = []
        scene_data[scene].append(item)
    
    random.seed(seed)
    
    train_data = []
    val_data = []
    test_data = []
    
    for scene, items in scene_data.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train+n_val])
        test_data.extend(items[n_train+n_val:])
    
    # 保存分割后的数据
    output_dir = os.path.dirname(annotation_file)
    
    train_file = os.path.join(output_dir, 'train_split.json')
    val_file = os.path.join(output_dir, 'val_split.json')
    test_file = os.path.join(output_dir, 'test_split.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据集划分完成:")
    print(f"  训练集: {len(train_data)} 样本 -> {train_file}")
    print(f"  验证集: {len(val_data)} 样本 -> {val_file}")
    print(f"  测试集: {len(test_data)} 样本 -> {test_file}")
    
    return train_file, val_file, test_file


def get_train_transform():
    """训练时的数据增强"""
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
    ])


def get_val_transform():
    """验证时不增强"""
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])


# ============================================
# 损失函数1：ITC (Image-Text Contrastive)
# ============================================
def compute_itc_loss(image_feats, text_feats, temp, device):
    """
    图文对比学习损失（InfoNCE）
    
    Args:
        image_feats: [B, num_queries, embed_dim] 或 [B, embed_dim]
        text_feats: [B, embed_dim]
        temp: 温度参数
        device: 设备
    
    Returns:
        loss: ITC 损失
        acc_i2t: 图->文检索准确率
        acc_t2i: 文->图检索准确率
    """
    # 1. 处理 image_feats 维度
    if image_feats.dim() == 3:
        image_feats = image_feats.mean(dim=1)
    
    # 2. L2 归一化
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    
    # 3. 计算相似度矩阵
    sim_i2t = image_feats @ text_feats.t() / temp
    sim_t2i = sim_i2t.t()
    
    # 4. 标签
    labels = torch.arange(image_feats.size(0), device=device)
    
    # 5. 双向交叉熵损失
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    loss = (loss_i2t + loss_t2i) / 2
    
    # 6. 计算准确率
    with torch.no_grad():
        acc_i2t = (sim_i2t.argmax(dim=1) == labels).float().mean()
        acc_t2i = (sim_t2i.argmax(dim=1) == labels).float().mean()
    
    return loss, acc_i2t.item(), acc_t2i.item()


# ============================================
# 损失函数2：Scene Classification
# ============================================
class SceneClassificationHead(nn.Module):
    """场景分类头"""
    def __init__(self, input_dim=256, num_classes=10, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, embed_dim] 图像特征
        Returns:
            logits: [B, num_classes]
        """
        return self.classifier(x)


def compute_scene_loss(scene_head, image_feats, scene_labels, device):
    """
    场景分类损失
    
    Args:
        scene_head: 场景分类头
        image_feats: [B, num_queries, embed_dim] 或 [B, embed_dim]
        scene_labels: [B] 场景标签
        device: 设备
    
    Returns:
        loss: 分类损失
        acc: 分类准确率
    """
    # 1. 处理 image_feats 维度
    if image_feats.dim() == 3:
        image_feats = image_feats.mean(dim=1)  # [B, embed_dim]
    
    # 2. 分类预测
    logits = scene_head(image_feats)  # [B, num_classes]
    
    # 3. 计算损失
    loss = F.cross_entropy(logits, scene_labels)
    
    # 4. 计算准确率
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == scene_labels).float().mean()
    
    return loss, acc.item()


# ============================================
# 模型加载
# ============================================
def load_base_model(checkpoint_path):
    """加载 BLIP2 基础模型"""
    print(f"\n📥 加载基础模型...")
    
    model = Blip2Qformer(
        vit_model="clip_L",
        img_size=224,
        freeze_vit=True,
        num_query_token=32,
        embed_dim=256,
        max_txt_len=77,  # 增加文本长度以适应详细描述
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get("model", checkpoint)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"✅ 加载预训练权重: {checkpoint_path}")
        if missing_keys:
            print(f"⚠️ 缺失的键 ({len(missing_keys)}): {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"⚠️ 多余的键 ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
    else:
        print(f"⚠️ 未找到权重文件: {checkpoint_path}，使用随机初始化")
    
    return model


def setup_lora(model, lora_config):
    """为 Q-Former 添加 LoRA 适配器"""
    print(f"\n🔧 配置 LoRA...")
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=lora_config['rank'],
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        target_modules=lora_config['target_modules'],
        bias="none",
    )
    
    model.Qformer = get_peft_model(model.Qformer, peft_config)
    
    print(f"  Rank: {lora_config['rank']}")
    print(f"  Alpha: {lora_config['alpha']}")
    print(f"  Target modules: {lora_config['target_modules']}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n💾 参数统计:")
    print(f"  可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  总参数: {total_params:,}")
    
    return model


def get_lora_config(profile='balanced'):
    """预定义的 LoRA 配置"""
    configs = {
        'minimal': {
            'rank': 4,
            'alpha': 8,
            'dropout': 0.1,
            'target_modules': ['query', 'value'],
        },
        'balanced': {
            'rank': 8,
            'alpha': 16,
            'dropout': 0.05,
            'target_modules': ['query', 'key', 'value'],
        },
        'full': {
            'rank': 16,
            'alpha': 32,
            'dropout': 0.05,
            'target_modules': ['query', 'key', 'value', 'dense'],
        }
    }
    return configs.get(profile, configs['balanced'])


# ============================================
# Early Stopping
# ============================================
class EarlyStopping:
    """Early Stopping 机制"""
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ============================================
# 训练一个 Epoch（ITC + Scene）
# ============================================
def train_one_epoch(model, scene_head, train_loader, optimizer, scheduler, scaler, 
                     device, epoch, writer, global_step, loss_weights):
    """训练一个 epoch（使用 ITC + Scene 联合损失）"""
    model.train()
    model.visual_encoder.eval()  # 保持 ViT 冻结
    scene_head.train()
    
    total_loss = 0
    total_itc_loss = 0
    total_scene_loss = 0
    total_acc_i2t = 0
    total_acc_t2i = 0
    total_scene_acc = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, samples in enumerate(pbar):
        images = samples['image'].to(device)
        texts = samples['text']
        scene_labels = samples['scene_label'].to(device)
        
        with autocast():
            # ========== ITC Loss ==========
            image_feats, text_feats = model({'image': images, 'text': texts})
            
            itc_loss, acc_i2t, acc_t2i = compute_itc_loss(
                image_feats, text_feats, model.temp, device
            )
            
            # ========== Scene Classification Loss ==========
            scene_loss, scene_acc = compute_scene_loss(
                scene_head, image_feats, scene_labels, device
            )
            
            # ========== 总损失 ==========
            loss = (
                loss_weights['itc'] * itc_loss + 
                loss_weights['scene'] * scene_loss
            )
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        total_itc_loss += itc_loss.item()
        total_scene_loss += scene_loss.item()
        total_acc_i2t += acc_i2t
        total_acc_t2i += acc_t2i
        total_scene_acc += scene_acc
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'itc': f'{itc_loss.item():.4f}',
            'scene': f'{scene_loss.item():.4f}',
            'i2t': f'{acc_i2t:.2%}',
            'scene_acc': f'{scene_acc:.2%}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # 记录到 TensorBoard
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/total_loss', loss.item(), global_step[0])
            writer.add_scalar('Train/itc_loss', itc_loss.item(), global_step[0])
            writer.add_scalar('Train/scene_loss', scene_loss.item(), global_step[0])
            writer.add_scalar('Train/acc_i2t', acc_i2t, global_step[0])
            writer.add_scalar('Train/acc_t2i', acc_t2i, global_step[0])
            writer.add_scalar('Train/scene_acc', scene_acc, global_step[0])
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step[0])
        
        global_step[0] += 1
    
    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'itc_loss': total_itc_loss / n,
        'scene_loss': total_scene_loss / n,
        'acc_i2t': total_acc_i2t / n,
        'acc_t2i': total_acc_t2i / n,
        'scene_acc': total_scene_acc / n,
    }


# ============================================
# 验证（ITC + Scene）
# ============================================
def validate(model, scene_head, val_loader, device, epoch, writer, loss_weights):
    """验证（使用 ITC + Scene）"""
    model.eval()
    scene_head.eval()
    
    total_loss = 0
    total_itc_loss = 0
    total_scene_loss = 0
    total_acc_i2t = 0
    total_acc_t2i = 0
    total_scene_acc = 0
    
    # 用于计算每个类别的准确率
    scene_correct = torch.zeros(NUM_SCENE_CLASSES)
    scene_total = torch.zeros(NUM_SCENE_CLASSES)
    
    with torch.no_grad():
        for samples in tqdm(val_loader, desc="Validating"):
            images = samples['image'].to(device)
            texts = samples['text']
            scene_labels = samples['scene_label'].to(device)
            
            with autocast():
                # ITC
                image_feats, text_feats = model({'image': images, 'text': texts})
                itc_loss, acc_i2t, acc_t2i = compute_itc_loss(
                    image_feats, text_feats, model.temp, device
                )
                
                # Scene
                scene_loss, scene_acc = compute_scene_loss(
                    scene_head, image_feats, scene_labels, device
                )
                
                loss = loss_weights['itc'] * itc_loss + loss_weights['scene'] * scene_loss
            
            total_loss += loss.item()
            total_itc_loss += itc_loss.item()
            total_scene_loss += scene_loss.item()
            total_acc_i2t += acc_i2t
            total_acc_t2i += acc_t2i
            total_scene_acc += scene_acc
            
            # 统计每个类别的准确率
            if image_feats.dim() == 3:
                image_feats_2d = image_feats.mean(dim=1)
            else:
                image_feats_2d = image_feats
            logits = scene_head(image_feats_2d)
            pred = logits.argmax(dim=1)
            
            for i in range(len(scene_labels)):
                label = scene_labels[i].item()
                scene_total[label] += 1
                if pred[i] == scene_labels[i]:
                    scene_correct[label] += 1
    
    n = len(val_loader)
    results = {
        'loss': total_loss / n,
        'itc_loss': total_itc_loss / n,
        'scene_loss': total_scene_loss / n,
        'acc_i2t': total_acc_i2t / n,
        'acc_t2i': total_acc_t2i / n,
        'scene_acc': total_scene_acc / n,
    }
    
    # 记录到 TensorBoard
    writer.add_scalar('Val/total_loss', results['loss'], epoch)
    writer.add_scalar('Val/itc_loss', results['itc_loss'], epoch)
    writer.add_scalar('Val/scene_loss', results['scene_loss'], epoch)
    writer.add_scalar('Val/acc_i2t', results['acc_i2t'], epoch)
    writer.add_scalar('Val/scene_acc', results['scene_acc'], epoch)
    
    print(f"\n📊 验证结果:")
    print(f"  Total Loss: {results['loss']:.4f}")
    print(f"  ITC Loss: {results['itc_loss']:.4f} | I2T Acc: {results['acc_i2t']:.2%}")
    print(f"  Scene Loss: {results['scene_loss']:.4f} | Scene Acc: {results['scene_acc']:.2%}")
    
    # 打印每个场景的准确率
    print(f"\n📈 各场景分类准确率:")
    for i in range(NUM_SCENE_CLASSES):
        if scene_total[i] > 0:
            acc = scene_correct[i] / scene_total[i]
            print(f"  {ID_TO_SCENE[i]:8s}: {acc:.2%} ({int(scene_correct[i])}/{int(scene_total[i])})")
    
    return results


# ============================================
# 保存检查点
# ============================================
def save_checkpoint(model, scene_head, optimizer, scheduler, epoch, loss, output_dir, is_best=False):
    """保存检查点"""
    os.makedirs(output_dir, exist_ok=True)
    
    lora_dir = os.path.join(output_dir, f'checkpoint_epoch_{epoch}')
    os.makedirs(lora_dir, exist_ok=True)
    
    # 保存 LoRA 权重
    model.Qformer.save_pretrained(lora_dir)
    
    # 保存场景分类头
    torch.save(scene_head.state_dict(), os.path.join(lora_dir, 'scene_head.pth'))
    
    # 保存训练状态
    state = {
        'epoch': epoch,
        'loss': loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, os.path.join(lora_dir, 'training_state.pth'))
    
    print(f"✅ 保存检查点: {lora_dir}")
    
    if is_best:
        best_dir = os.path.join(output_dir, 'best_model')
        os.makedirs(best_dir, exist_ok=True)
        model.Qformer.save_pretrained(best_dir)
        torch.save(scene_head.state_dict(), os.path.join(best_dir, 'scene_head.pth'))
        torch.save(state, os.path.join(best_dir, 'training_state.pth'))
        print(f"🏆 保存最佳模型: {best_dir}")


# ============================================
# 主训练函数
# ============================================
def main():
    config = {
        # 数据路径
        'annotation_file': '/workspace/vlm/lab/output/scene_annotations.json',
        'image_dir': '/data/fasion/train/image',
        
        # 数据划分（首次运行会自动划分）
        'train_file': '/workspace/vlm/lab/output/train_split.json',
        'val_file': '/workspace/vlm/lab/output/val_split.json',
        'test_file': '/workspace/vlm/lab/output/test_split.json',
        
        # 训练参数
        'batch_size': 32,
        'num_workers': 8,
        'use_key_features': False,  # True: 使用 key_features, False: 使用 text
        
        # 模型
        'checkpoint': 'checkpoint_04.pth',
        'lora_profile': 'full',  # minimal / balanced / full
        
        # 训练
        'epochs': 30,
        'lr': 1e-4,
        'weight_decay': 0.05,
        'warmup_epochs': 2,
        
        # 损失权重
        'loss_weights': {
            'itc': 0.2,    # ITC 损失权重
            'scene': 0.8,  # 场景分类损失权重
        },
        
        # Early stopping
        'patience': 30,
        'min_delta': 0.0,
        
        # 输出
        'output_dir': 'outputs/fashion_lora_itc_scene',
        'log_dir': 'runs/fashion_lora_itc_scene',
        
        'device': 'cuda:4' ,
    }
    
    print("="*60)
    print("🚀 BLIP2 Q-Former LoRA 微调 - 时尚场景分类")
    print("="*60)
    print(f"\n⚙️ 配置:")
    for key, value in config.items():
        if not key.endswith('_file') and not key.endswith('_dir'):
            print(f"  {key}: {value}")
    
    # 检查数据文件
    if not os.path.exists(config['annotation_file']):
        raise FileNotFoundError(f"❌ 标注文件不存在: {config['annotation_file']}")
    
    if not os.path.exists(config['image_dir']):
        raise FileNotFoundError(f"❌ 图片目录不存在: {config['image_dir']}")
    
    # 数据划分（如果分割文件不存在）
    if not all(os.path.exists(f) for f in [config['train_file'], config['val_file'], config['test_file']]):
        train_file, val_file, test_file = split_dataset(
            config['annotation_file'],
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )
        config['train_file'] = train_file
        config['val_file'] = val_file
        config['test_file'] = test_file
    
    # 准备数据集
    print(f"\n📊 加载数据集...")
    print(f"  使用文本字段: {'key_features' if config['use_key_features'] else 'text'}")
    
    train_dataset = FashionSceneDataset(
        config['train_file'],
        config['image_dir'],
        transform=get_train_transform(),
        use_key_features=config['use_key_features']
    )
    
    val_dataset = FashionSceneDataset(
        config['val_file'],
        config['image_dir'],
        transform=get_val_transform(),
        use_key_features=config['use_key_features']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'], 
        pin_memory=True,
        collate_fn=lambda x: {
            'image': torch.stack([item['image'] for item in x]),
            'text': [item['text'] for item in x],
            'scene_label': torch.tensor([item['scene_label'] for item in x], dtype=torch.long),
        }
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'], 
        pin_memory=True,
        collate_fn=lambda x: {
            'image': torch.stack([item['image'] for item in x]),
            'text': [item['text'] for item in x],
            'scene_label': torch.tensor([item['scene_label'] for item in x], dtype=torch.long),
        }
    )
    
    # 准备模型
    model = load_base_model(config['checkpoint'])
    lora_config = get_lora_config(config['lora_profile'])
    model = setup_lora(model, lora_config)
    model.to(config['device'])
    
    # 场景分类头
    scene_head = SceneClassificationHead(
        input_dim=256,
        num_classes=NUM_SCENE_CLASSES,
        dropout=0.1
    ).to(config['device'])
    
    print(f"\n🎯 场景分类头:")
    print(f"  输入维度: 256")
    print(f"  类别数: {NUM_SCENE_CLASSES}")
    print(f"  参数量: {sum(p.numel() for p in scene_head.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) + 
        list(scene_head.parameters()),
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config['warmup_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    # 训练
    print(f"\n🏋️ 开始训练...")
    print(f"  总 Epochs: {config['epochs']}")
    print(f"  训练批次: {len(train_loader)} batches/epoch")
    print(f"  验证批次: {len(val_loader)} batches/epoch")
    
    global_step = [0]
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_results = train_one_epoch(
            model, scene_head, train_loader, optimizer, scheduler, scaler,
            config['device'], epoch, writer, global_step, config['loss_weights']
        )
        
        print(f"\n📈 训练结果:")
        print(f"  Total Loss: {train_results['loss']:.4f}")
        print(f"  ITC Loss: {train_results['itc_loss']:.4f} | I2T: {train_results['acc_i2t']:.2%}")
        print(f"  Scene Loss: {train_results['scene_loss']:.4f} | Acc: {train_results['scene_acc']:.2%}")
        
        val_results = validate(
            model, scene_head, val_loader, config['device'], epoch, writer, config['loss_weights']
        )
        
        is_best = early_stopping(val_results['loss'])
        if is_best:
            print(f"🎉 新的最佳模型! (Val Loss: {val_results['loss']:.4f})")
            best_val_loss = val_results['loss']
        
        if epoch % 5 == 0 or is_best:
            save_checkpoint(
                model, scene_head, optimizer, scheduler, epoch, 
                val_results['loss'], config['output_dir'], is_best
            )
        
        if early_stopping.early_stop:
            print(f"\n⏹️ Early Stopping 触发! (连续 {config['patience']} 个 epoch 无改善)")
            break
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 训练完成!")
    print(f"  总时长: {elapsed_time/3600:.2f} 小时")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"\n📁 输出文件:")
    print(f"  模型权重: {config['output_dir']}")
    print(f"  训练日志: {config['log_dir']}")
    print(f"\n📊 查看日志:")
    print(f"  tensorboard --logdir={config['log_dir']}")
    print(f'  "$BROWSER" http://localhost:6006')
    
    writer.close()


if __name__ == "__main__":
    main()