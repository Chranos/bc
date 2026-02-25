#!/usr/bin/env python3
"""
通用场景分类微调脚本
支持微调以下模型：
- CLIP-ViT-B/32
- CLIP-ViT-L/14
- ResNet-50
- ViT-Base
- 其他分类模型

只使用场景分类损失（CrossEntropy），不涉及图文对比学习
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
from torchvision import transforms, models
from tqdm import tqdm
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import argparse  # 添加这个


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
class SceneClassificationDataset(Dataset):
    """场景分类数据集"""
    
    def __init__(self, annotation_file: str, image_dir: str, transform=None):
        print(f"📂 加载数据集: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform or self._default_transform()
        
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
        
        # 统计分布
        scene_counts = {}
        for item in self.data:
            scene_id = item['scene_id']
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        
        print(f"\n📊 场景分布:")
        for scene_id in sorted(scene_counts.keys()):
            count = scene_counts[scene_id]
            ratio = count / len(self.data) * 100
            print(f"  {ID_TO_SCENE[scene_id]:8s}: {count:4d} ({ratio:5.1f}%)")
    
    def _default_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
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
            print(f"⚠️ 加载失败: {image_path}, {e}")
            image = torch.zeros(3, 224, 224)
        
        return {
            'image': image,
            'label': item['scene_id'],
            'file_name': item['file_name'],
        }


def get_train_transform():
    """训练数据增强"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
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
    """验证不增强"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])


# ============================================
# 模型定义
# ============================================
class CLIPClassifier(nn.Module):
    """CLIP 分类器（冻结编码器 + 新分类头）"""
    
    def __init__(self, clip_model_name='ViT-B/32', num_classes=10, freeze_encoder=True):
        super().__init__()
        print(f"📥 加载 CLIP: {clip_model_name}")
        
        import clip
        self.clip_model, self.preprocess = clip.load(clip_model_name, device='cpu')
        
        # 冻结 CLIP 编码器
        if freeze_encoder:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # 获取特征维度
        if 'ViT-B' in clip_model_name:
            feature_dim = 512
        elif 'ViT-L' in clip_model_name:
            feature_dim = 768
        elif 'RN50' in clip_model_name:
            feature_dim = 1024
        else:
            feature_dim = 512
        
        # 新的分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"  特征维度: {feature_dim}")
        print(f"  编码器冻结: {freeze_encoder}")
    
    def forward(self, images):
        # CLIP 图像编码
        with torch.no_grad() if next(self.clip_model.parameters()).requires_grad == False else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features.float()
        
        # 分类
        logits = self.classifier(image_features)
        return logits


class ResNetClassifier(nn.Module):
    """ResNet 分类器（冻结部分层 + 新分类头）"""
    
    def __init__(self, num_classes=10, freeze_backbone=True, pretrained=True):
        super().__init__()
        print(f"📥 加载 ResNet-50 (pretrained={pretrained})")
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 冻结前面的层
        if freeze_backbone:
            # 只微调最后的 layer4 和 fc
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        
        # 替换分类头
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"  特征维度: {feature_dim}")
        print(f"  部分冻结: {freeze_backbone}")
    
    def forward(self, images):
        return self.backbone(images)


class ViTClassifier(nn.Module):
    """ViT 分类器（冻结部分层 + 新分类头）"""
    
    def __init__(self, model_name='vit_base_patch16_224', num_classes=10, 
                 freeze_backbone=True, pretrained=True):
        super().__init__()
        print(f"📥 加载 ViT: {model_name} (pretrained={pretrained})")
        
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # 冻结前面的层
        if freeze_backbone:
            # 只微调最后几层
            total_blocks = len(self.backbone.blocks)
            for i, block in enumerate(self.backbone.blocks):
                if i < total_blocks - 3:  # 冻结前面的块
                    for param in block.parameters():
                        param.requires_grad = False
        
        # 新分类头
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"  特征维度: {feature_dim}")
        print(f"  部分冻结: {freeze_backbone}")
    
    def forward(self, images):
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits


def create_model(model_name: str, num_classes=10, freeze_encoder=True, pretrained=True):
    """
    创建模型
    
    Args:
        model_name: 模型名称
            - 'clip-vit-b32': CLIP ViT-B/32
            - 'clip-vit-l14': CLIP ViT-L/14
            - 'resnet50': ResNet-50
            - 'vit-base': ViT-Base
        num_classes: 分类类别数
        freeze_encoder: 是否冻结编码器
        pretrained: 是否使用预训练权重
    
    Returns:
        model: 分类模型
    """
    if model_name == 'clip-vit-b32':
        return CLIPClassifier('ViT-B/32', num_classes, freeze_encoder)
    elif model_name == 'clip-vit-l14':
        return CLIPClassifier('ViT-L/14', num_classes, freeze_encoder)
    elif model_name == 'resnet50':
        return ResNetClassifier(num_classes, freeze_encoder, pretrained)
    elif model_name == 'vit-base':
        return ViTClassifier('vit_base_patch16_224', num_classes, freeze_encoder, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================
# 训练一个 Epoch
# ============================================
def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, writer, global_step):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, samples in enumerate(pbar):
        images = samples['image'].to(device)
        labels = samples['label'].to(device)
        
        with autocast():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += len(labels)
        
        acc = total_correct / total_samples
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2%}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # TensorBoard
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/loss', loss.item(), global_step[0])
            writer.add_scalar('Train/acc', acc, global_step[0])
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step[0])
        
        global_step[0] += 1
    
    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'acc': total_correct / total_samples,
    }


# ============================================
# 验证
# ============================================
@torch.no_grad()
def validate(model, val_loader, device, epoch, writer):
    """验证"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # 每个类别的统计
    class_correct = torch.zeros(NUM_SCENE_CLASSES)
    class_total = torch.zeros(NUM_SCENE_CLASSES)
    
    for samples in tqdm(val_loader, desc="Validating"):
        images = samples['image'].to(device)
        labels = samples['label'].to(device)
        
        with autocast():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item()
        
        pred = logits.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 统计每个类别
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if pred[i] == labels[i]:
                class_correct[label] += 1
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    
    results = {
        'loss': total_loss / len(val_loader),
        'acc': acc,
    }
    
    # TensorBoard
    writer.add_scalar('Val/loss', results['loss'], epoch)
    writer.add_scalar('Val/acc', results['acc'], epoch)
    
    print(f"\n📊 验证结果:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['acc']:.2%}")
    
    # 各类别准确率
    print(f"\n📈 各类别准确率:")
    for i in range(NUM_SCENE_CLASSES):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"  {ID_TO_SCENE[i]:8s}: {class_acc:.2%} ({int(class_correct[i])}/{int(class_total[i])})")
    
    return results


# ============================================
# 保存模型
# ============================================
def save_checkpoint(model, optimizer, scheduler, epoch, acc, output_dir, model_name, is_best=False):
    """保存检查点"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(output_dir, f'{model_name}_epoch_{epoch}.pth')
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'acc': acc,
    }
    
    torch.save(state, checkpoint_path)
    print(f"✅ 保存检查点: {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(output_dir, f'{model_name}_best.pth')
        torch.save(state, best_path)
        print(f"🏆 保存最佳模型: {best_path}")


# ============================================
# Early Stopping
# ============================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
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
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
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
# 主函数
# ============================================
def main():
    # ========== 添加命令行参数解析 ==========
    parser = argparse.ArgumentParser(description='通用场景分类微调脚本')
    parser.add_argument('--model_name', type=str, default='clip-vit-b32',
                       choices=['clip-vit-b32', 'clip-vit-l14', 'resnet50', 'vit-base'],
                       help='模型名称')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda:4', help='设备')
    parser.add_argument('--freeze_encoder', action='store_true', default=True, help='冻结编码器')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # ========== 配置（使用命令行参数） ==========
    config = {
        # 数据
        'train_file': '/workspace/vlm/lab/output/train_split.json',
        'val_file': '/workspace/vlm/lab/output/val_split.json',
        'image_dir': '/data/fasion/train/image',
        
        # 模型（从命令行获取）
        'model_name': args.model_name,
        'freeze_encoder': args.freeze_encoder,
        'pretrained': True,
        
        # 训练（从命令行获取）
        'batch_size': args.batch_size,
        'num_workers': 8,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 0.01,
        'warmup_epochs': 2,
        
        # Early stopping
        'patience': args.patience,
        'min_delta': 0.001,
        
        # 输出（根据模型名自动设置）
        'output_dir': f'outputs/finetuned_{args.model_name}',
        'log_dir': f'runs/finetune_{args.model_name}',
        
        # 设备（从命令行获取）
        'device': args.device,
    }
    
    print("="*60)
    print(f"🚀 微调场景分类模型: {config['model_name']}")
    print("="*60)
    print(f"\n⚙️  训练配置:")
    print(f"  模型: {config['model_name']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  学习率: {config['lr']}")
    print(f"  设备: {config['device']}")
    print(f"  冻结编码器: {config['freeze_encoder']}")
    
    # ========== 数据集 ==========
    train_dataset = SceneClassificationDataset(
        config['train_file'],
        config['image_dir'],
        transform=get_train_transform()
    )
    
    val_dataset = SceneClassificationDataset(
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
            'label': torch.tensor([item['label'] for item in x], dtype=torch.long),
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
            'label': torch.tensor([item['label'] for item in x], dtype=torch.long),
        }
    )
    
    # ========== 模型 ==========
    model = create_model(
        config['model_name'],
        num_classes=NUM_SCENE_CLASSES,
        freeze_encoder=config['freeze_encoder'],
        pretrained=config['pretrained']
    )
    model.to(config['device'])
    
    # 统计参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n💾 参数统计:")
    print(f"  可训练: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  总参数: {total_params:,}")
    
    # ========== 优化器 ==========
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config['warmup_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'], mode='max')
    
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    # ========== 训练 ==========
    print(f"\n🏋️ 开始训练...")
    
    global_step = [0]
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_results = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config['device'], epoch, writer, global_step
        )
        
        print(f"\n📈 训练结果:")
        print(f"  Loss: {train_results['loss']:.4f}")
        print(f"  Accuracy: {train_results['acc']:.2%}")
        
        val_results = validate(model, val_loader, config['device'], epoch, writer)
        
        is_best = early_stopping(val_results['acc'])
        if is_best:
            print(f"🎉 新的最佳模型! (Acc: {val_results['acc']:.2%})")
            best_acc = val_results['acc']
        
        if epoch % 5 == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_results['acc'],
                config['output_dir'], config['model_name'], is_best
            )
        
        if early_stopping.early_stop:
            print(f"\n⏹️ Early Stopping! (连续 {config['patience']} epoch 无改善)")
            break
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 训练完成!")
    print(f"  总时长: {elapsed_time/3600:.2f} 小时")
    print(f"  最佳准确率: {best_acc:.2%}")
    print(f"\n📁 输出文件:")
    print(f"  模型权重: {config['output_dir']}")
    print(f"  训练日志: {config['log_dir']}")
    print(f"\n📊 查看日志:")
    print(f'  tensorboard --logdir={config["log_dir"]}')
    
    writer.close()


if __name__ == "__main__":
    main()