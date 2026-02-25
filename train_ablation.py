#!/usr/bin/env python3
"""
BLIP2 损失函数消融实验脚本

支持以下损失组合：
1. Only ITC (Image-Text Contrastive)
2. Only Scene Classification
3. ITC + Scene (不同权重比例)
4. ITC + ITM (Image-Text Matching)
5. ITC + Scene + ITM

用于验证不同损失函数对模型性能的影响
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
import argparse

from peft import LoraConfig, get_peft_model, TaskType
from models.blip2_qformer import Blip2Qformer


# ============================================
# 场景类别定义
# ============================================
SCENE_CATEGORIES = [
    '职场正装', '职场休闲', '运动健身', '户外探险', '居家休闲',
    '社交聚会', '旅行度假', '运动赛事', '婚礼相关', '特殊功能',
]
SCENE_TO_ID = {name: idx for idx, name in enumerate(SCENE_CATEGORIES)}
ID_TO_SCENE = {idx: name for idx, name in enumerate(SCENE_CATEGORIES)}
NUM_SCENE_CLASSES = len(SCENE_CATEGORIES)


# ============================================
# 数据集
# ============================================
class FashionSceneDataset(Dataset):
    """时尚场景分类数据集"""
    
    def __init__(self, annotation_file, image_dir, transform=None, use_key_features=False):
        print(f"📂 加载数据集: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform or self._default_transform()
        self.use_key_features = use_key_features
        
        # 验证数据
        valid_data = []
        for item in self.data:
            if 'file_name' not in item or 'scene_category' not in item:
                continue
            
            scene = item['scene_category']
            if scene not in SCENE_TO_ID:
                continue
            
            image_path = os.path.join(self.image_dir, item['file_name'])
            if not os.path.exists(image_path):
                continue
            
            item['scene_id'] = SCENE_TO_ID[scene]
            valid_data.append(item)
        
        self.data = valid_data
        print(f"✅ 有效样本: {len(self.data)}")
    
    def _default_transform(self):
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
        
        image_path = os.path.join(self.image_dir, item['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            image = torch.zeros(3, 224, 224)
        
        # 文本处理
        if self.use_key_features and 'key_features' in item:
            text = item['key_features']
        else:
            text_content = item.get('text', '')
            scene_category = item.get('scene_category', '')
            text = f"{text_content}，适合{scene_category}" if text_content and scene_category else (text_content or scene_category)
        
        return {
            'image': image,
            'text': text,
            'scene_label': item['scene_id'],
        }


def get_train_transform():
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ])


def get_val_transform():
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
# 损失函数定义
# ============================================

# 1. ITC Loss (Image-Text Contrastive)
def compute_itc_loss(image_feats, text_feats, temp, device):
    """图文对比学习损失"""
    if image_feats.dim() == 3:
        image_feats = image_feats.mean(dim=1)
    
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    
    sim_i2t = image_feats @ text_feats.t() / temp
    sim_t2i = sim_i2t.t()
    
    labels = torch.arange(image_feats.size(0), device=device)
    
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    loss = (loss_i2t + loss_t2i) / 2
    
    with torch.no_grad():
        acc_i2t = (sim_i2t.argmax(dim=1) == labels).float().mean().item()
        acc_t2i = (sim_t2i.argmax(dim=1) == labels).float().mean().item()
    
    return loss, acc_i2t, acc_t2i


# 2. Scene Classification Loss
class SceneClassificationHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=10, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


def compute_scene_loss(scene_head, image_feats, scene_labels):
    if image_feats.dim() == 3:
        image_feats = image_feats.mean(dim=1)
    
    logits = scene_head(image_feats)
    loss = F.cross_entropy(logits, scene_labels)
    
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == scene_labels).float().mean().item()
    
    return loss, acc


# 3. ITM Loss (Image-Text Matching)
class ITMHead(nn.Module):
    """图文匹配头（二分类：匹配/不匹配）"""
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # 二分类
        )
    
    def forward(self, multimodal_feats):
        """
        Args:
            multimodal_feats: [B, embed_dim] 融合后的多模态特征
        Returns:
            logits: [B, 2]
        """
        return self.classifier(multimodal_feats)


def compute_itm_loss(model, itm_head, images, texts, device):
    """
    图文匹配损失
    
    构造正负样本对：
    - 正样本：原始的图文对
    - 负样本：随机打乱文本
    """
    batch_size = images.size(0)
    
    # 1. 正样本：原始配对
    with torch.no_grad():
        # 提取图文特征（用于构造融合特征）
        image_feats, text_feats = model({'image': images, 'text': texts})
        if image_feats.dim() == 3:
            image_feats = image_feats.mean(dim=1)
    
    # 简单融合：element-wise 相加
    pos_feats = image_feats + text_feats
    
    # 2. 负样本：打乱文本
    neg_indices = torch.randperm(batch_size)
    neg_texts = [texts[i] for i in neg_indices]
    
    with torch.no_grad():
        _, neg_text_feats = model({'image': images, 'text': neg_texts})
    
    neg_feats = image_feats + neg_text_feats
    
    # 3. 拼接正负样本
    all_feats = torch.cat([pos_feats, neg_feats], dim=0)
    labels = torch.cat([
        torch.ones(batch_size, dtype=torch.long, device=device),   # 正样本标签=1
        torch.zeros(batch_size, dtype=torch.long, device=device)   # 负样本标签=0
    ])
    
    # 4. 计算损失
    logits = itm_head(all_feats)
    loss = F.cross_entropy(logits, labels)
    
    # 5. 计算准确率
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean().item()
    
    return loss, acc


# ============================================
# 消融实验配置
# ============================================
ABLATION_CONFIGS = {
    # ========== 单一损失 ==========
    'itc_only': {
        'name': 'ITC Only',
        'losses': ['itc'],
        'weights': {'itc': 1.0},
        'description': '仅使用图文对比学习损失',
    },
    'scene_only': {
        'name': 'Scene Only',
        'losses': ['scene'],
        'weights': {'scene': 1.0},
        'description': '仅使用场景分类损失',
    },
    
    # ========== 双损失组合 ==========
    'itc_scene_equal': {
        'name': 'ITC + Scene (1:1)',
        'losses': ['itc', 'scene'],
        'weights': {'itc': 0.5, 'scene': 0.5},
        'description': 'ITC 和 Scene 等权重',
    },
    'itc_scene_2_8': {
        'name': 'ITC + Scene (2:8)',
        'losses': ['itc', 'scene'],
        'weights': {'itc': 0.2, 'scene': 0.8},
        'description': 'ITC:Scene = 2:8 (Scene 为主)',
    },
    'itc_scene_8_2': {
        'name': 'ITC + Scene (8:2)',
        'losses': ['itc', 'scene'],
        'weights': {'itc': 0.8, 'scene': 0.2},
        'description': 'ITC:Scene = 8:2 (ITC 为主)',
    },
    
    # ========== ITM 相关 ==========
    'itc_itm': {
        'name': 'ITC + ITM',
        'losses': ['itc', 'itm'],
        'weights': {'itc': 0.5, 'itm': 0.5},
        'description': '图文对比 + 图文匹配',
    },
    
    # ========== 三损失组合 ==========
    'itc_scene_itm': {
        'name': 'ITC + Scene + ITM',
        'losses': ['itc', 'scene', 'itm'],
        'weights': {'itc': 0.3, 'scene': 0.5, 'itm': 0.2},
        'description': '三种损失联合训练',
    },
    'itc_scene_itm_equal': {
        'name': 'ITC + Scene + ITM (Equal)',
        'losses': ['itc', 'scene', 'itm'],
        'weights': {'itc': 0.33, 'scene': 0.33, 'itm': 0.34},
        'description': '三种损失等权重',
    },
}


def get_ablation_config(name):
    """获取消融实验配置"""
    if name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation config: {name}. Available: {list(ABLATION_CONFIGS.keys())}")
    return ABLATION_CONFIGS[name]


# ============================================
# 模型初始化
# ============================================
def load_base_model(checkpoint_path):
    """加载基础模型"""
    model = Blip2Qformer(
        vit_model="clip_L",
        img_size=224,
        freeze_vit=True,
        num_query_token=32,
        embed_dim=256,
        max_txt_len=77,
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 加载预训练权重: {checkpoint_path}")
    
    return model


def setup_lora(model, rank=8, alpha=16):
    """添加 LoRA"""
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=['query', 'key', 'value'],
        bias="none",
    )
    
    model.Qformer = get_peft_model(model.Qformer, peft_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"💾 可训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


# ============================================
# 训练一个 Epoch
# ============================================
def train_one_epoch(model, heads, train_loader, optimizer, scheduler, scaler,
                     device, epoch, writer, global_step, config):
    """训练一个 epoch（支持不同的损失组合）"""
    model.train()
    model.visual_encoder.eval()
    
    # 设置各个头为训练模式
    for head in heads.values():
        if head is not None:
            head.train()
    
    total_metrics = {
        'loss': 0,
        'itc_loss': 0, 'itc_acc_i2t': 0, 'itc_acc_t2i': 0,
        'scene_loss': 0, 'scene_acc': 0,
        'itm_loss': 0, 'itm_acc': 0,
    }
    
    ablation_cfg = config['ablation']
    losses_to_use = ablation_cfg['losses']
    weights = ablation_cfg['weights']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, samples in enumerate(pbar):
        images = samples['image'].to(device)
        texts = samples['text']
        scene_labels = samples['scene_label'].to(device)
        
        with autocast():
            # 提取特征
            image_feats, text_feats = model({'image': images, 'text': texts})
            
            total_loss = 0
            loss_dict = {}
            
            # ========== ITC Loss ==========
            if 'itc' in losses_to_use:
                itc_loss, acc_i2t, acc_t2i = compute_itc_loss(
                    image_feats, text_feats, model.temp, device
                )
                total_loss += weights['itc'] * itc_loss
                loss_dict['itc'] = itc_loss.item()
                total_metrics['itc_loss'] += itc_loss.item()
                total_metrics['itc_acc_i2t'] += acc_i2t
                total_metrics['itc_acc_t2i'] += acc_t2i
            
            # ========== Scene Loss ==========
            if 'scene' in losses_to_use:
                scene_loss, scene_acc = compute_scene_loss(
                    heads['scene'], image_feats, scene_labels
                )
                total_loss += weights['scene'] * scene_loss
                loss_dict['scene'] = scene_loss.item()
                total_metrics['scene_loss'] += scene_loss.item()
                total_metrics['scene_acc'] += scene_acc
            
            # ========== ITM Loss ==========
            if 'itm' in losses_to_use:
                itm_loss, itm_acc = compute_itm_loss(
                    model, heads['itm'], images, texts, device
                )
                total_loss += weights['itm'] * itm_loss
                loss_dict['itm'] = itm_loss.item()
                total_metrics['itm_loss'] += itm_loss.item()
                total_metrics['itm_acc'] += itm_acc
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_metrics['loss'] += total_loss.item()
        
        # 更新进度条
        pbar_info = {'loss': f'{total_loss.item():.4f}'}
        pbar_info.update({k: f'{v:.4f}' for k, v in loss_dict.items()})
        pbar.set_postfix(pbar_info)
        
        # TensorBoard
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/total_loss', total_loss.item(), global_step[0])
            for name, value in loss_dict.items():
                writer.add_scalar(f'Train/{name}_loss', value, global_step[0])
        
        global_step[0] += 1
    
    # 计算平均
    n = len(train_loader)
    return {k: v / n for k, v in total_metrics.items()}


# ============================================
# 验证
# ============================================
@torch.no_grad()
def validate(model, heads, val_loader, device, epoch, writer, config):
    """验证"""
    model.eval()
    for head in heads.values():
        if head is not None:
            head.eval()
    
    total_metrics = {
        'loss': 0,
        'itc_loss': 0, 'itc_acc_i2t': 0,
        'scene_loss': 0, 'scene_acc': 0,
        'itm_loss': 0, 'itm_acc': 0,
    }
    
    ablation_cfg = config['ablation']
    losses_to_use = ablation_cfg['losses']
    weights = ablation_cfg['weights']
    
    for samples in tqdm(val_loader, desc="Validating"):
        images = samples['image'].to(device)
        texts = samples['text']
        scene_labels = samples['scene_label'].to(device)
        
        with autocast():
            image_feats, text_feats = model({'image': images, 'text': texts})
            
            total_loss = 0
            
            if 'itc' in losses_to_use:
                itc_loss, acc_i2t, acc_t2i = compute_itc_loss(
                    image_feats, text_feats, model.temp, device
                )
                total_loss += weights['itc'] * itc_loss
                total_metrics['itc_loss'] += itc_loss.item()
                total_metrics['itc_acc_i2t'] += acc_i2t
            
            if 'scene' in losses_to_use:
                scene_loss, scene_acc = compute_scene_loss(
                    heads['scene'], image_feats, scene_labels
                )
                total_loss += weights['scene'] * scene_loss
                total_metrics['scene_loss'] += scene_loss.item()
                total_metrics['scene_acc'] += scene_acc
            
            if 'itm' in losses_to_use:
                itm_loss, itm_acc = compute_itm_loss(
                    model, heads['itm'], images, texts, device
                )
                total_loss += weights['itm'] * itm_loss
                total_metrics['itm_loss'] += itm_loss.item()
                total_metrics['itm_acc'] += itm_acc
            
            total_metrics['loss'] += total_loss.item()
    
    n = len(val_loader)
    results = {k: v / n for k, v in total_metrics.items()}
    
    # TensorBoard
    writer.add_scalar('Val/total_loss', results['loss'], epoch)
    if 'itc' in losses_to_use:
        writer.add_scalar('Val/itc_acc', results['itc_acc_i2t'], epoch)
    if 'scene' in losses_to_use:
        writer.add_scalar('Val/scene_acc', results['scene_acc'], epoch)
    if 'itm' in losses_to_use:
        writer.add_scalar('Val/itm_acc', results['itm_acc'], epoch)
    
    print(f"\n📊 验证结果:")
    print(f"  Total Loss: {results['loss']:.4f}")
    if 'itc' in losses_to_use:
        print(f"  ITC: Loss={results['itc_loss']:.4f}, I2T Acc={results['itc_acc_i2t']:.2%}")
    if 'scene' in losses_to_use:
        print(f"  Scene: Loss={results['scene_loss']:.4f}, Acc={results['scene_acc']:.2%}")
    if 'itm' in losses_to_use:
        print(f"  ITM: Loss={results['itm_loss']:.4f}, Acc={results['itm_acc']:.2%}")
    
    return results


# ============================================
# 保存检查点
# ============================================
def save_checkpoint(model, heads, optimizer, epoch, results, output_dir, is_best=False):
    """保存检查点"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(output_dir, f'checkpoint_epoch_{epoch}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存 LoRA
    model.Qformer.save_pretrained(checkpoint_dir)
    
    # 保存各个头
    for name, head in heads.items():
        if head is not None:
            torch.save(head.state_dict(), os.path.join(checkpoint_dir, f'{name}_head.pth'))
    
    # 保存训练状态
    state = {
        'epoch': epoch,
        'results': results,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(checkpoint_dir, 'training_state.pth'))
    
    print(f"✅ 保存检查点: {checkpoint_dir}")
    
    if is_best:
        best_dir = os.path.join(output_dir, 'best_model')
        os.makedirs(best_dir, exist_ok=True)
        model.Qformer.save_pretrained(best_dir)
        for name, head in heads.items():
            if head is not None:
                torch.save(head.state_dict(), os.path.join(best_dir, f'{name}_head.pth'))
        torch.save(state, os.path.join(best_dir, 'training_state.pth'))
        print(f"🏆 保存最佳模型: {best_dir}")


# ============================================
# 主函数
# ============================================
def main():
    parser = argparse.ArgumentParser(description='BLIP2 损失函数消融实验')
    parser.add_argument('--ablation', type=str, default='itc_scene_2_8',
                       choices=list(ABLATION_CONFIGS.keys()),
                       help='消融实验配置')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda:4', help='设备')
    args = parser.parse_args()
    
    # 获取消融配置
    ablation_cfg = get_ablation_config(args.ablation)
    
    config = {
        'train_file': '/workspace/vlm/lab/output/train_split.json',
        'val_file': '/workspace/vlm/lab/output/val_split.json',
        'image_dir': '/data/fasion/train/image',
        
        'checkpoint': 'checkpoint_04.pth',
        
        'batch_size': args.batch_size,
        'num_workers': 8,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 0.05,
        
        'ablation': ablation_cfg,
        
        'output_dir': f'outputs/ablation_{args.ablation}',
        'log_dir': f'runs/ablation_{args.ablation}',
        
        'device': args.device,
    }
    
    print("="*60)
    print(f"🔬 BLIP2 损失函数消融实验")
    print("="*60)
    print(f"\n📋 实验配置: {ablation_cfg['name']}")
    print(f"  描述: {ablation_cfg['description']}")
    print(f"  损失函数: {ablation_cfg['losses']}")
    print(f"  权重: {ablation_cfg['weights']}")
    
    # 加载数据
    train_dataset = FashionSceneDataset(
        config['train_file'],
        config['image_dir'],
        transform=get_train_transform()
    )
    
    val_dataset = FashionSceneDataset(
        config['val_file'],
        config['image_dir'],
        transform=get_val_transform()
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
    
    # 初始化模型
    model = load_base_model(config['checkpoint'])
    model = setup_lora(model)
    model.to(config['device'])
    
    # 初始化各个头（根据需要）
    heads = {}
    
    if 'scene' in ablation_cfg['losses']:
        heads['scene'] = SceneClassificationHead(256, NUM_SCENE_CLASSES).to(config['device'])
        print(f"✅ 初始化场景分类头")
    else:
        heads['scene'] = None
    
    if 'itm' in ablation_cfg['losses']:
        heads['itm'] = ITMHead(256, 512).to(config['device'])
        print(f"✅ 初始化图文匹配头")
    else:
        heads['itm'] = None
    
    # 优化器
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    for head in heads.values():
        if head is not None:
            params += list(head.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    # 训练
    print(f"\n🏋️ 开始训练...")
    global_step = [0]
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_results = train_one_epoch(
            model, heads, train_loader, optimizer, scheduler, scaler,
            config['device'], epoch, writer, global_step, config
        )
        
        val_results = validate(
            model, heads, val_loader, config['device'], epoch, writer, config
        )
        
        is_best = val_results['loss'] < best_val_loss
        if is_best:
            print(f"🎉 新的最佳模型! (Loss: {val_results['loss']:.4f})")
            best_val_loss = val_results['loss']
        
        if epoch % 5 == 0 or is_best:
            save_checkpoint(model, heads, optimizer, epoch, val_results, config['output_dir'], is_best)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 训练完成!")
    print(f"  时长: {elapsed/3600:.2f} 小时")
    print(f"  最佳损失: {best_val_loss:.4f}")
    print(f"\n📁 输出: {config['output_dir']}")
    print(f"📊 日志: tensorboard --logdir={config['log_dir']}")
    
    writer.close()


if __name__ == "__main__":
    main()