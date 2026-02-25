#!/usr/bin/env python3
"""
BLIP2 + LoRA 图文检索 + 场景分类 综合测试脚本
使用场景类别名称作为文本进行检索和分类评估
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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from peft import PeftModel
from models.blip2_qformer import Blip2Qformer


# ============================================
# 场景类别定义（与训练保持一致）
# ============================================
SCENE_CATEGORIES = [
    '职场正装', '职场休闲', '运动健身', '户外探险', '居家休闲',
    '社交聚会', '旅行度假', '运动赛事', '婚礼相关', '特殊功能',
]
SCENE_TO_ID = {name: idx for idx, name in enumerate(SCENE_CATEGORIES)}
ID_TO_SCENE = {idx: name for idx, name in enumerate(SCENE_CATEGORIES)}
NUM_SCENE_CLASSES = len(SCENE_CATEGORIES)


# ============================================
# 场景分类头（与训练保持一致）
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
        return self.classifier(x)


# ============================================
# 测试数据集（使用场景类别名称作为文本）
# ============================================
class TestDataset(Dataset):
    """图文检索 + 场景分类测试数据集 - 使用 text + "，适合" + scene_category"""
    def __init__(self, annotation_file, image_dir, transform=None, use_key_features=False):
        print(f"📂 加载测试数据: {annotation_file}")
        
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
            
            # 验证场景标签
            scene = item['scene_category']
            if scene not in SCENE_TO_ID:
                continue
            
            # 验证图片存在
            image_path = os.path.join(self.image_dir, item['file_name'])
            if not os.path.exists(image_path):
                continue
            
            item['scene_id'] = SCENE_TO_ID[scene]
            
            # ===== 修改：拼接 text 和 scene_category =====
            if self.use_key_features and 'key_features' in item:
                # 使用 key_features
                item['combined_text'] = item['key_features']
            else:
                # 拼接 text + "，适合" + scene_category
                text_content = item.get('text', '')
                scene_category = item.get('scene_category', '')
                
                if text_content and scene_category:
                    item['combined_text'] = f"{text_content}，适合{scene_category}"
                elif text_content:
                    item['combined_text'] = text_content
                else:
                    item['combined_text'] = scene_category
            
            valid_data.append(item)
        
        self.data = valid_data
        print(f"✅ 加载 {len(self.data)} 个有效测试样本")
        print(f"💡 使用文本格式: text + \"，适合\" + scene_category")
        
        # 打印几个示例
        print(f"\n📝 文本示例:")
        for i in range(min(3, len(self.data))):
            print(f"  样本 {i+1}: {self.data[i]['combined_text'][:100]}...")
        
        # 统计场景分布
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
        
        return {
            'image': image,
            'text': item['combined_text'],  # 使用拼接后的文本
            'scene_label': item['scene_id'],
            'file_name': item['file_name'],
            'idx': idx,
        }

# ============================================
# 模型加载
# ============================================
def load_model_with_lora(base_checkpoint, lora_checkpoint, scene_head_path, device):
    """
    加载基础模型 + LoRA + 场景分类头
    
    Args:
        base_checkpoint: 基础 BLIP2 权重路径
        lora_checkpoint: LoRA 适配器权重目录
        scene_head_path: 场景分类头权重路径
        device: 设备
    
    Returns:
        model: 加载了 LoRA 的模型
        scene_head: 场景分类头
    """
    print(f"\n📥 加载模型...")
    print(f"  基础权重: {base_checkpoint}")
    print(f"  LoRA 权重: {lora_checkpoint}")
    print(f"  分类头权重: {scene_head_path}")
    
    # 1. 加载基础模型
    model = Blip2Qformer(
        vit_model="clip_L",
        img_size=224,
        freeze_vit=True,
        num_query_token=32,
        embed_dim=256,
        max_txt_len=77,
    )
    
    if os.path.exists(base_checkpoint):
        checkpoint = torch.load(base_checkpoint, map_location='cpu')
        state_dict = checkpoint.get("model", checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"✅ 加载基础权重")
    else:
        print(f"⚠️ 未找到基础权重: {base_checkpoint}")
    
    # 2. 加载 LoRA 适配器
    if os.path.exists(lora_checkpoint):
        print(f"🔧 加载 LoRA 适配器...")
        model.Qformer = PeftModel.from_pretrained(
            model.Qformer,
            lora_checkpoint,
            is_trainable=False
        )
        print(f"✅ LoRA 权重加载成功")
    else:
        raise FileNotFoundError(f"❌ 未找到 LoRA 权重: {lora_checkpoint}")
    
    model.to(device)
    model.eval()
    
    # 3. 加载场景分类头
    scene_head = SceneClassificationHead(
        input_dim=256,
        num_classes=NUM_SCENE_CLASSES,
        dropout=0.1
    ).to(device)
    
    if os.path.exists(scene_head_path):
        scene_head.load_state_dict(torch.load(scene_head_path, map_location=device))
        print(f"✅ 场景分类头加载成功")
    else:
        raise FileNotFoundError(f"❌ 未找到分类头权重: {scene_head_path}")
    
    scene_head.eval()
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n💾 模型参数: {total_params:,}")
    
    return model, scene_head


# ============================================
# 特征提取 + 分类预测
# ============================================
@torch.no_grad()
def extract_features_and_predict(model, scene_head, dataloader, device):
    """
    提取图像/文本特征 + 场景分类预测
    
    Returns:
        image_feats: [N, embed_dim]
        text_feats: [N, embed_dim]
        scene_labels: [N] 真实标签
        scene_preds: [N] 预测标签
        scene_probs: [N, num_classes] 预测概率
        indices: [N] 样本索引
    """
    print(f"\n🔍 提取特征并预测...")
    
    image_feats_list = []
    text_feats_list = []
    scene_labels_list = []
    scene_preds_list = []
    scene_probs_list = []
    indices_list = []
    
    for batch in tqdm(dataloader, desc="处理批次"):
        images = batch['image'].to(device)
        texts = batch['text']
        scene_labels = batch['scene_label']
        indices = batch['idx']
        
        # 提取特征
        image_feats, text_feats = model({'image': images, 'text': texts})
        
        # 处理 image_feats 维度
        if image_feats.dim() == 3:
            image_feats = image_feats.mean(dim=1)
        
        # 场景分类预测
        logits = scene_head(image_feats)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        # L2 归一化（用于检索）
        image_feats_norm = F.normalize(image_feats, dim=-1)
        text_feats_norm = F.normalize(text_feats, dim=-1)
        
        # 收集结果
        image_feats_list.append(image_feats_norm.cpu())
        text_feats_list.append(text_feats_norm.cpu())
        scene_labels_list.append(scene_labels)
        scene_preds_list.append(preds.cpu())
        scene_probs_list.append(probs.cpu())
        indices_list.extend(indices.tolist())
    
    # 合并所有结果
    image_feats = torch.cat(image_feats_list, dim=0)
    text_feats = torch.cat(text_feats_list, dim=0)
    scene_labels = torch.cat(scene_labels_list, dim=0)
    scene_preds = torch.cat(scene_preds_list, dim=0)
    scene_probs = torch.cat(scene_probs_list, dim=0)
    
    print(f"✅ 处理完成")
    print(f"  图像特征: {image_feats.shape}")
    print(f"  文本特征: {text_feats.shape}")
    print(f"  场景标签: {scene_labels.shape}")
    
    return image_feats, text_feats, scene_labels, scene_preds, scene_probs, indices_list


# ============================================
# 图文检索评估（场景级别）
# ============================================
def evaluate_scene_based_retrieval(image_feats, text_feats, scene_labels, top_k=[1, 5, 10]):
    """
    基于场景的图文检索评估
    由于每个场景类别有多个样本，评估时考虑同类别样本的检索性能
    文本格式: text + "，适合" + scene_category
    """
    print(f"\n" + "="*60)
    print(f"📊 基于场景的图文检索评估")
    print(f"💡 评估指标: 能否检索到相同场景类别的样本")
    print(f"💡 文本格式: text + \"，适合\" + scene_category")
    print(f"="*60)
    
    N = image_feats.size(0)
    sim_matrix = image_feats @ text_feats.t()
    
    metrics = {}
    
    # ========== Image-to-Text (场景级别) ==========
    print(f"\n📷➡️📝 图像检索文本 (同场景视为正确):")
    ranks = []
    for i in range(N):
        query_scene = scene_labels[i].item()
        sims = sim_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)
        
        # 找到第一个同场景样本的排名
        for rank, idx in enumerate(sorted_indices, 1):
            if scene_labels[idx].item() == query_scene:
                ranks.append(rank)
                break
    
    ranks = np.array(ranks)
    for k in top_k:
        recall = (ranks <= k).mean() * 100
        metrics[f'i2t_R@{k}'] = recall
        print(f"  R@{k:2d}: {recall:6.2f}%")
    
    metrics['i2t_median_rank'] = float(np.median(ranks))
    metrics['i2t_mean_rank'] = float(np.mean(ranks))
    print(f"  Median Rank: {metrics['i2t_median_rank']:.1f}")
    print(f"  Mean Rank: {metrics['i2t_mean_rank']:.1f}")
    
    # ========== Text-to-Image (场景级别) ==========
    print(f"\n📝➡️📷 文本检索图像 (同场景视为正确):")
    sim_matrix_t = sim_matrix.t()
    ranks = []
    for i in range(N):
        query_scene = scene_labels[i].item()
        sims = sim_matrix_t[i]
        sorted_indices = torch.argsort(sims, descending=True)
        
        # 找到第一个同场景样本的排名
        for rank, idx in enumerate(sorted_indices, 1):
            if scene_labels[idx].item() == query_scene:
                ranks.append(rank)
                break
    
    ranks = np.array(ranks)
    for k in top_k:
        recall = (ranks <= k).mean() * 100
        metrics[f't2i_R@{k}'] = recall
        print(f"  R@{k:2d}: {recall:6.2f}%")
    
    metrics['t2i_median_rank'] = float(np.median(ranks))
    metrics['t2i_mean_rank'] = float(np.mean(ranks))
    print(f"  Median Rank: {metrics['t2i_median_rank']:.1f}")
    print(f"  Mean Rank: {metrics['t2i_mean_rank']:.1f}")
    
    # ========== 平均指标 ==========
    print(f"\n📈 平均检索性能:")
    for k in top_k:
        avg_recall = (metrics[f'i2t_R@{k}'] + metrics[f't2i_R@{k}']) / 2
        metrics[f'avg_R@{k}'] = avg_recall
        print(f"  Avg R@{k:2d}: {avg_recall:6.2f}%")
    
    return metrics


# ============================================
# 场景分类评估
# ============================================
def evaluate_classification(scene_labels, scene_preds, scene_probs):
    """计算场景分类指标"""
    print(f"\n" + "="*60)
    print(f"🎯 场景分类评估")
    print(f"="*60)
    
    scene_labels = scene_labels.numpy()
    scene_preds = scene_preds.numpy()
    
    # 总体准确率
    accuracy = (scene_labels == scene_preds).mean() * 100
    print(f"\n整体准确率: {accuracy:.2f}%")
    
    # 详细分类报告
    print(f"\n📋 详细分类报告:")
    report = classification_report(
        scene_labels,
        scene_preds,
        target_names=SCENE_CATEGORIES,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(scene_labels, scene_preds)
    
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
    }
    
    # 每个类别的准确率
    print(f"\n📊 各场景准确率:")
    class_acc = {}
    for i, scene_name in enumerate(SCENE_CATEGORIES):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            class_acc[scene_name] = float(acc)
            print(f"  {scene_name:8s}: {acc:6.2f}% ({cm[i, i]}/{cm[i].sum()})")
        else:
            class_acc[scene_name] = 0.0
    
    metrics['class_accuracy'] = class_acc
    
    return metrics, cm


# ============================================
# 可视化混淆矩阵
# ============================================
def plot_confusion_matrix(cm, output_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(12, 10))
    
    # 归一化到 [0, 1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=SCENE_CATEGORIES,
        yticklabels=SCENE_CATEGORIES,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Scene Classification Confusion Matrix', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 混淆矩阵已保存: {output_path}")
    plt.close()


# ============================================
# 保存结果
# ============================================
def save_results(retrieval_metrics, classification_metrics, output_file):
    """保存所有评估结果"""
    results = {
        'retrieval': retrieval_metrics,
        'classification': classification_metrics,
        'text_format': 'text + "，适合" + scene_category',
        'note': '使用完整描述文本+场景类别进行检索评估'
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 完整结果已保存: {output_file}")


# ============================================
# 主函数
# ============================================
def main():
    config = {
        # 数据
        'test_file': '/workspace/vlm/lab/output/test_split.json',
        'image_dir': '/data/fasion/train/image',
        'use_key_features': False,  # 是否使用 key_features（False 则使用 text + scene_category）
        
        # 模型
        'base_checkpoint': 'checkpoint_04.pth',
        'lora_checkpoint': 'outputs/fashion_lora_itc_scene/best_model',
        'scene_head_path': 'outputs/fashion_lora_itc_scene/best_model/scene_head.pth',
        
        # 评估
        'batch_size': 64,
        'num_workers': 8,
        'top_k': [1, 5, 10, 20],
        
        # 输出
        'output_dir': '/workspace/vlm/lab/output',
        'results_file': '/workspace/vlm/lab/output/test_results_text_scene.json',
        'confusion_matrix_file': '/workspace/vlm/lab/output/confusion_matrix_text_scene.png',
        
        'device': 'cuda:4',
    }
    
    print("="*60)
    print("🔬 BLIP2 + LoRA 综合测试")
    print("  - 图文检索 (text + \"，适合\" + scene_category)")
    print("  - 场景分类 (Scene Classification)")
    print("="*60)
    
    # 检查文件
    for key in ['test_file', 'image_dir', 'base_checkpoint', 'lora_checkpoint', 'scene_head_path']:
        path = config[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 路径不存在: {path}")
    
    # 加载数据集
    test_dataset = TestDataset(
        config['test_file'], 
        config['image_dir'],
        use_key_features=config['use_key_features']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=lambda x: {
            'image': torch.stack([item['image'] for item in x]),
            'text': [item['text'] for item in x],
            'scene_label': torch.tensor([item['scene_label'] for item in x]),
            'idx': torch.tensor([item['idx'] for item in x]),
        }
    )
    
    # 加载模型
    model, scene_head = load_model_with_lora(
        config['base_checkpoint'],
        config['lora_checkpoint'],
        config['scene_head_path'],
        config['device']
    )
    
    # 提取特征 + 预测
    image_feats, text_feats, scene_labels, scene_preds, scene_probs, indices = \
        extract_features_and_predict(model, scene_head, test_loader, config['device'])
    
    # ========== 评估检索（场景级别） ==========
    retrieval_metrics = evaluate_scene_based_retrieval(
        image_feats, text_feats, scene_labels, top_k=config['top_k']
    )
    
    # ========== 评估分类 ==========
    classification_metrics, cm = evaluate_classification(
        scene_labels, scene_preds, scene_probs
    )
    
    # ========== 可视化混淆矩阵 ==========
    plot_confusion_matrix(cm, config['confusion_matrix_file'])
    
    # ========== 保存结果 ==========
    save_results(retrieval_metrics, classification_metrics, config['results_file'])
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("✅ 测试完成!")
    print("="*60)
    print(f"\n📊 关键指标:")
    print(f"  场景检索 Avg R@1:  {retrieval_metrics['avg_R@1']:.2f}%")
    print(f"  场景检索 Avg R@5:  {retrieval_metrics['avg_R@5']:.2f}%")
    print(f"  场景分类准确率:    {classification_metrics['accuracy']:.2f}%")
    print(f"\n📁 输出文件:")
    print(f"  结果JSON: {config['results_file']}")
    print(f"  混淆矩阵: {config['confusion_matrix_file']}")


if __name__ == "__main__":
    main()

