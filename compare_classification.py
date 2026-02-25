#!/usr/bin/env python3
"""
场景分类准确率对比实验脚本
支持对比多个模型的分类性能，包括：
- BLIP2 (未微调基线 - 零样本)
- BLIP2 + LoRA (微调后)
- CLIP 系列 (零样本 & 微调)
- ResNet-50 (预训练 & 微调)
- ViT-Base (预训练 & 微调)

输出详细的对比表格、混淆矩阵和可视化图表
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

from models.model_zoo import create_model


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
# 生成场景描述文本（用于零样本分类）
# ============================================
def generate_scene_prompts(template_style='descriptive'):
    """
    生成场景类别的文本描述
    
    Args:
        template_style: 模板风格
            - 'simple': 简单类别名
            - 'descriptive': 描述性文本
            - 'context': 带场景上下文
    
    Returns:
        prompts: 文本描述列表
    """
    if template_style == 'simple':
        # 简单类别名
        prompts = SCENE_CATEGORIES
    
    elif template_style == 'descriptive':
        # 描述性文本
        prompts = [
            '适合职场正式场合的正装服饰',
            '适合职场的商务休闲服装',
            '适合运动健身的运动服装',
            '适合户外探险的功能性服装',
            '适合家中穿着的居家休闲服',
            '适合社交聚会的时尚服装',
            '适合旅行度假的轻便服装',
            '适合运动赛事的专业装备',
            '适合婚礼场合的礼服',
            '具有特殊功能的服装',
        ]
    
    elif template_style == 'context':
        # 带场景上下文
        prompts = [
            '一张展示职场正装服饰的图片',
            '一张展示职场休闲服装的图片',
            '一张展示运动健身服装的图片',
            '一张展示户外探险服装的图片',
            '一张展示居家休闲服的图片',
            '一张展示社交聚会服装的图片',
            '一张展示旅行度假服装的图片',
            '一张展示运动赛事装备的图片',
            '一张展示婚礼礼服的图片',
            '一张展示特殊功能服装的图片',
        ]
    
    else:
        raise ValueError(f"Unknown template style: {template_style}")
    
    print(f"  📝 场景描述模板: {template_style}")
    print(f"  示例: {prompts[0]}")
    
    return prompts


# ============================================
# 零样本分类器（用于未微调的模型）
# ============================================
class ZeroShotClassifier:
    """
    零样本分类器 - 用于没有分类头的模型
    通过图文相似度进行分类
    """
    def __init__(self, model, scene_prompts: List[str], device='cuda'):
        """
        Args:
            model: 模型包装器
            scene_prompts: 场景类别的文本描述
            device: 设备
        """
        self.model = model
        self.device = device
        
        # 预计算场景文本特征
        print(f"  🔄 预计算场景文本特征...")
        
        # 创建一个 dummy 图像
        dummy_images = [Image.new('RGB', (224, 224), color=(128, 128, 128))] * len(scene_prompts)
        
        try:
            _, text_feats = model.extract_features(dummy_images, scene_prompts)
            
            if text_feats is None:
                raise ValueError("模型不支持文本特征提取")
            
            self.scene_text_feats = F.normalize(text_feats, dim=-1)
            print(f"  ✅ 场景文本特征形状: {self.scene_text_feats.shape}")
        
        except Exception as e:
            print(f"  ❌ 文本特征提取失败: {e}")
            raise
    
    @torch.no_grad()
    def predict(self, images):
        """
        零样本预测
        
        Args:
            images: PIL Images 列表
        
        Returns:
            logits: [B, num_classes]
        """
        # 提取图像特征
        image_feats, _ = self.model.extract_features(images, [""] * len(images))
        
        # 归一化
        image_feats = F.normalize(image_feats, dim=-1)
        
        # 计算相似度作为 logits
        logits = image_feats @ self.scene_text_feats.t()
        
        # 缩放到更合理的范围
        logits = logits * 100.0
        
        return logits


# ============================================
# 扩展的模型包装器（支持零样本分类）
# ============================================
class ExtendedModelWrapper:
    """
    扩展的模型包装器 - 为没有分类头的模型添加零样本分类能力
    """
    def __init__(self, base_model, use_zero_shot=False, scene_prompts=None):
        """
        Args:
            base_model: 原始模型包装器
            use_zero_shot: 是否使用零样本分类
            scene_prompts: 场景类别的文本描述（用于零样本）
        """
        self.base_model = base_model
        self.use_zero_shot = use_zero_shot
        self.zero_shot_classifier = None
        
        if use_zero_shot and scene_prompts:
            print(f"  🎯 初始化零样本分类器...")
            try:
                self.zero_shot_classifier = ZeroShotClassifier(
                    base_model, 
                    scene_prompts,
                    base_model.device
                )
            except Exception as e:
                print(f"  ⚠️ 零样本分类器初始化失败: {e}")
                self.zero_shot_classifier = None
    
    def extract_features(self, images, texts):
        """提取特征"""
        return self.base_model.extract_features(images, texts)
    
    def classify_scene(self, images):
        """场景分类"""
        # 优先使用原生分类头
        try:
            native_logits = self.base_model.classify_scene(images)
            if native_logits is not None:
                return native_logits
        except Exception as e:
            print(f"  ⚠️ 原生分类失败: {e}")
        
        # 否则使用零样本分类
        if self.use_zero_shot and self.zero_shot_classifier:
            return self.zero_shot_classifier.predict(images)
        
        return None
    
    def get_model_info(self):
        """获取模型信息"""
        info = self.base_model.get_model_info()
        if self.use_zero_shot and self.zero_shot_classifier:
            info['classification_method'] = 'zero-shot'
        elif 'classification_method' not in info:
            info['classification_method'] = 'supervised'
        return info
    
    @property
    def model_name(self):
        return getattr(self.base_model, 'model_name', 'Unknown')
    
    @property
    def device(self):
        return self.base_model.device


# ============================================
# 测试数据集
# ============================================
class ClassificationTestDataset(Dataset):
    """场景分类测试数据集"""
    
    def __init__(self, annotation_file: str, image_dir: str, transform=None):
        """
        Args:
            annotation_file: 标注文件路径
            image_dir: 图片目录
            transform: 图像变换（可选）
        """
        print(f"📂 加载测试数据: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform or self._default_transform()
        
        # 验证和过滤数据
        valid_data = []
        for item in self.data:
            # 检查必需字段
            if 'file_name' not in item or 'scene_category' not in item:
                continue
            
            # 验证场景类别
            scene = item['scene_category']
            if scene not in SCENE_TO_ID:
                continue
            
            # 验证图片存在
            image_path = os.path.join(self.image_dir, item['file_name'])
            if not os.path.exists(image_path):
                continue
            
            item['scene_id'] = SCENE_TO_ID[scene]
            valid_data.append(item)
        
        self.data = valid_data
        print(f"✅ 加载 {len(self.data)} 个有效样本")
        
        # 统计场景分布
        self._print_distribution()
    
    def _default_transform(self):
        """默认图像变换"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    
    def _print_distribution(self):
        """打印场景分布"""
        scene_counts = {}
        for item in self.data:
            scene_id = item['scene_id']
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        
        print(f"\n📊 场景分布:")
        for scene_id in sorted(scene_counts.keys()):
            count = scene_counts[scene_id]
            ratio = count / len(self.data) * 100
            scene_name = ID_TO_SCENE[scene_id]
            print(f"  {scene_name:8s}: {count:4d} ({ratio:5.1f}%)")
    
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
            print(f"⚠️ 加载图像失败: {image_path}, {e}")
            image = torch.zeros(3, 224, 224)
        
        return {
            'image': image,
            'scene_label': item['scene_id'],
            'file_name': item['file_name'],
        }


# ============================================
# 单模型评估
# ============================================
@torch.no_grad()
def evaluate_single_model(model, test_loader, device, model_name):
    """
    评估单个模型的分类性能
    
    Args:
        model: 模型包装器
        test_loader: 测试数据加载器
        device: 设备
        model_name: 模型名称
    
    Returns:
        results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"🔬 评估模型: {model_name}")
    print(f"{'='*60}")
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    # 逐批次预测
    for batch in tqdm(test_loader, desc=f"推理中"):
        images = batch['image'].to(device)
        labels = batch['scene_label']
        
        # 转换为 PIL Images
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        pil_images = [to_pil(img.cpu()) for img in images]
        
        try:
            # 获取分类 logits
            logits = model.classify_scene(pil_images)
            
            if logits is None:
                raise ValueError(f"{model_name} 不支持场景分类")
            
            # 计算预测
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_labels.append(labels)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
        
        except Exception as e:
            print(f"⚠️ 处理批次时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_labels) == 0:
        print(f"❌ 没有成功处理的批次")
        return None
    
    # 合并结果
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    # ========== 计算指标 ==========
    
    # 1. 总体准确率
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # 2. 每类别的精确率、召回率、F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # 3. 加权平均指标
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 4. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 5. Top-5 准确率
    top5_acc = 0.0
    if NUM_SCENE_CLASSES >= 5:
        top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
        top5_correct = np.array([label in top5_preds[i] for i, label in enumerate(all_labels)])
        top5_acc = top5_correct.mean() * 100
    
    # ========== 打印结果 ==========
    print(f"\n📊 分类结果:")
    print(f"  整体准确率: {accuracy:.2f}%")
    if NUM_SCENE_CLASSES >= 5:
        print(f"  Top-5 准确率: {top5_acc:.2f}%")
    print(f"  加权精确率: {precision_avg*100:.2f}%")
    print(f"  加权召回率: {recall_avg*100:.2f}%")
    print(f"  加权 F1 分数: {f1_avg*100:.2f}%")
    
    print(f"\n📋 各类别指标:")
    print(f"{'场景':<10s} {'准确率':>8s} {'精确率':>8s} {'召回率':>8s} {'F1分数':>8s} {'样本数':>8s}")
    print("-" * 60)
    
    class_metrics = {}
    for i, scene_name in enumerate(SCENE_CATEGORIES):
        if support[i] > 0:
            class_acc = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0.0
            class_metrics[scene_name] = {
                'accuracy': float(class_acc),
                'precision': float(precision[i] * 100),
                'recall': float(recall[i] * 100),
                'f1': float(f1[i] * 100),
                'support': int(support[i]),
            }
            print(f"{scene_name:<10s} {class_acc:7.2f}% {precision[i]*100:7.2f}% "
                  f"{recall[i]*100:7.2f}% {f1[i]*100:7.2f}% {support[i]:7d}")
    
    # ========== 返回结果 ==========
    results = {
        'model_name': model_name,
        'overall': {
            'accuracy': float(accuracy),
            'top5_accuracy': float(top5_acc) if NUM_SCENE_CLASSES >= 5 else None,
            'weighted_precision': float(precision_avg * 100),
            'weighted_recall': float(recall_avg * 100),
            'weighted_f1': float(f1_avg * 100),
            'num_samples': len(all_labels),
        },
        'per_class': class_metrics,
        'confusion_matrix': cm.tolist(),
    }
    
    return results


# ============================================
# 可视化函数
# ============================================
def plot_accuracy_comparison(all_results: List[Dict], output_path: str):
    """绘制准确率对比柱状图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = [r['model_name'] for r in all_results]
    
    # 1. 整体准确率
    ax = axes[0, 0]
    accuracies = [r['overall']['accuracy'] for r in all_results]
    bars = ax.barh(model_names, accuracies, color='skyblue')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)
    
    # 2. Top-5 准确率
    ax = axes[0, 1]
    if all_results[0]['overall']['top5_accuracy'] is not None:
        top5_accs = [r['overall']['top5_accuracy'] for r in all_results]
        bars = ax.barh(model_names, top5_accs, color='lightgreen')
        ax.set_xlabel('Top-5 Accuracy (%)', fontsize=12)
        ax.set_title('Top-5 Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 100])
        for bar, acc in zip(bars, top5_accs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{acc:.2f}%', va='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Top-5 Accuracy N/A', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
    
    # 3. F1 分数
    ax = axes[1, 0]
    f1_scores = [r['overall']['weighted_f1'] for r in all_results]
    bars = ax.barh(model_names, f1_scores, color='lightcoral')
    ax.set_xlabel('Weighted F1 Score (%)', fontsize=12)
    ax.set_title('Weighted F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{f1:.2f}%', va='center', fontsize=10)
    
    # 4. 平均类别准确率
    ax = axes[1, 1]
    avg_class_accs = []
    for result in all_results:
        class_accs = [m['accuracy'] for m in result['per_class'].values()]
        avg_class_accs.append(np.mean(class_accs) if class_accs else 0.0)
    
    bars = ax.barh(model_names, avg_class_accs, color='plum')
    ax.set_xlabel('Average Per-Class Accuracy (%)', fontsize=12)
    ax.set_title('Average Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    for bar, acc in zip(bars, avg_class_accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图已保存: {output_path}")
    plt.close()


def plot_per_class_comparison(all_results: List[Dict], output_path: str):
    """绘制各类别准确率对比热力图"""
    model_names = [r['model_name'] for r in all_results]
    
    # 构建数据矩阵
    data = []
    for result in all_results:
        row = []
        for scene in SCENE_CATEGORIES:
            if scene in result['per_class']:
                row.append(result['per_class'][scene]['accuracy'])
            else:
                row.append(0.0)
        data.append(row)
    
    data = np.array(data)
    
    # 绘制热力图
    plt.figure(figsize=(14, max(6, len(model_names) * 0.6)))
    sns.heatmap(
        data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=SCENE_CATEGORIES,
        yticklabels=model_names,
        cbar_kws={'label': 'Accuracy (%)'},
        vmin=0,
        vmax=100,
        linewidths=0.5,
    )
    
    plt.xlabel('Scene Category', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 类别对比热力图已保存: {output_path}")
    plt.close()


def plot_confusion_matrices(all_results: List[Dict], output_dir: str):
    """为每个模型绘制混淆矩阵"""
    for result in all_results:
        model_name = result['model_name']
        cm = np.array(result['confusion_matrix'])
        
        # 归一化
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=SCENE_CATEGORIES,
            yticklabels=SCENE_CATEGORIES,
            cbar_kws={'label': 'Normalized Count'},
            vmin=0,
            vmax=1,
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        safe_name = model_name.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
        output_path = os.path.join(output_dir, f'confusion_matrix_{safe_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 {model_name} 混淆矩阵已保存")


def save_comparison_table(all_results: List[Dict], output_path: str):
    """保存对比表格为 CSV"""
    rows = []
    for result in all_results:
        row = {
            'Model': result['model_name'],
            'Accuracy (%)': f"{result['overall']['accuracy']:.2f}",
            'Top-5 Acc (%)': f"{result['overall'].get('top5_accuracy', 0):.2f}" if result['overall'].get('top5_accuracy') else 'N/A',
            'Weighted Precision (%)': f"{result['overall']['weighted_precision']:.2f}",
            'Weighted Recall (%)': f"{result['overall']['weighted_recall']:.2f}",
            'Weighted F1 (%)': f"{result['overall']['weighted_f1']:.2f}",
            'Num Samples': result['overall']['num_samples'],
        }
        
        # 添加各类别准确率
        for scene in SCENE_CATEGORIES:
            if scene in result['per_class']:
                row[f'{scene} Acc (%)'] = f"{result['per_class'][scene]['accuracy']:.2f}"
            else:
                row[f'{scene} Acc (%)'] = '0.00'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 对比表格已保存: {output_path}")
    
    # 打印表格预览
    print(f"\n📊 对比表格预览:")
    print(df[['Model', 'Accuracy (%)', 'Weighted F1 (%)']].to_string(index=False))


# ============================================
# 主函数
# ============================================
def main():
    """主函数 - 批量测试多个模型"""
    
    config = {
        # 数据
        'test_file': '/workspace/vlm/lab/output/test_split.json',
        'image_dir': '/data/fasion/train/image',
        'batch_size': 32,
        'num_workers': 4,
        
        # 设备
        'device': 'cuda:4',
        
        # 输出
        'output_dir': '/workspace/vlm/lab/output/classification_comparison',
        
        # 零样本配置
        'zero_shot_template': 'descriptive',  # 'simple', 'descriptive', 'context'
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("="*60)
    print("🔬 场景分类对比实验")
    print("="*60)
    
    # 生成场景描述（用于零样本分类）
    scene_prompts = generate_scene_prompts(config['zero_shot_template'])
    
    # ========== 定义要对比的模型 ==========
    models_to_compare = [
        # ========== 1. 零样本 CLIP 系列 ==========
        {
            'name': 'CLIP-ViT-B/32 (Zero-shot)',
            'model_name': 'clip-vit-b32',
            'kwargs': {'num_classes': 10},
            'use_zero_shot': True,
        },
        {
            'name': 'CLIP-ViT-L/14 (Zero-shot)',
            'model_name': 'clip-vit-l14',
            'kwargs': {'num_classes': 10},
            'use_zero_shot': True,
        },
        
        # ========== 2. 微调后的 CLIP 系列 ==========
        {
            'name': 'CLIP-ViT-B/32 (Finetuned)',
            'model_name': 'clip-vit-b32',
            'kwargs': {
                'checkpoint_path': 'outputs/finetuned_clip-vit-b32/clip-vit-b32_best.pth',
                'num_classes': 10,
            },
            'use_zero_shot': False,
        },
        {
            'name': 'CLIP-ViT-L/14 (Finetuned)',
            'model_name': 'clip-vit-l14',
            'kwargs': {
                'checkpoint_path': 'outputs/finetuned_clip-vit-l14/clip-vit-l14_best.pth',
                'num_classes': 10,
            },
            'use_zero_shot': False,
        },
        
        # ========== 3. ResNet-50 ==========
        {
            'name': 'ResNet-50 (Finetuned)',
            'model_name': 'resnet50',
            'kwargs': {
                'checkpoint_path': 'outputs/finetuned_resnet50/resnet50_best.pth',
                'num_classes': 10,
            },
            'use_zero_shot': False,
        },
        
        # ========== 4. ViT-Base ==========
        {
            'name': 'ViT-Base (Finetuned)',
            'model_name': 'vit-base',
            'kwargs': {
                'checkpoint_path': 'outputs/finetuned_vit-base/vit-base_best.pth',
                'num_classes': 10,
            },
            'use_zero_shot': False,
        },
        
        # ========== 5. BLIP2 + LoRA ==========
        {
            'name': 'BLIP2-LoRA (Ours)',
            'model_name': 'blip2-lora',
            'kwargs': {
                'base_checkpoint': 'checkpoint_04.pth',
                'lora_checkpoint': 'outputs/fashion_lora_itc_scene/best_model',
                'scene_head_path': 'outputs/fashion_lora_itc_scene/best_model/scene_head.pth',
            },
            'use_zero_shot': False,
        },
    ]
    
    # ========== 加载测试数据 ==========
    test_dataset = ClassificationTestDataset(
        config['test_file'],
        config['image_dir']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=lambda x: {
            'image': torch.stack([item['image'] for item in x]),
            'scene_label': torch.tensor([item['scene_label'] for item in x]),
        }
    )
    
    # ========== 评估所有模型 ==========
    all_results = []
    
    for model_config in models_to_compare:
        try:
            print(f"\n{'#'*60}")
            print(f"# {model_config['name']}")
            print(f"{'#'*60}")
            
            # 创建基础模型
            base_model = create_model(
                model_config['model_name'],
                device=config['device'],
                **model_config['kwargs']
            )
            
            # 如果需要零样本分类，包装模型
            if model_config.get('use_zero_shot', False):
                model = ExtendedModelWrapper(
                    base_model,
                    use_zero_shot=True,
                    scene_prompts=scene_prompts
                )
            else:
                model = base_model
            
            # 评估
            results = evaluate_single_model(
                model,
                test_loader,
                config['device'],
                model_config['name']
            )
            
            if results is not None:
                all_results.append(results)
                
                # 保存单个模型结果
                safe_name = model_config['name'].replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
                result_file = os.path.join(config['output_dir'], f"{safe_name}_results.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 释放内存
            del model
            if 'base_model' in locals():
                del base_model
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"❌ 评估 {model_config['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== 生成对比报告 ==========
    if len(all_results) == 0:
        print("❌ 没有成功评估的模型")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 生成对比报告...")
    print(f"{'='*60}")
    
    # 1. 保存汇总 JSON
    summary_file = os.path.join(config['output_dir'], 'comparison_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✅ 汇总结果已保存: {summary_file}")
    
    # 2. 保存对比表格
    table_file = os.path.join(config['output_dir'], 'comparison_table.csv')
    save_comparison_table(all_results, table_file)
    
    # 3. 绘制对比图
    comparison_plot = os.path.join(config['output_dir'], 'accuracy_comparison.png')
    plot_accuracy_comparison(all_results, comparison_plot)
    
    # 4. 绘制类别对比热力图
    heatmap_plot = os.path.join(config['output_dir'], 'per_class_comparison.png')
    plot_per_class_comparison(all_results, heatmap_plot)
    
    # 5. 绘制所有混淆矩阵
    print(f"\n📊 生成混淆矩阵...")
    plot_confusion_matrices(all_results, config['output_dir'])
    
    # ========== 最终总结 ==========
    print(f"\n{'='*60}")
    print(f"✅ 对比实验完成！")
    print(f"{'='*60}")
    
    print(f"\n🏆 排名 (按准确率):")
    sorted_results = sorted(all_results, key=lambda x: x['overall']['accuracy'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. {result['model_name']:<35s}: {result['overall']['accuracy']:.2f}%")
    
    print(f"\n📁 所有结果已保存到: {config['output_dir']}")


if __name__ == "__main__":
    main()