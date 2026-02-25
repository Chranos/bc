#!/usr/bin/env python3
"""
对比所有消融实验的场景分类准确率
在测试集上评估每个消融实验模型的性能
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 导入模型
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
# 零样本分类器（用于没有场景分类头的模型）
# ============================================
class ZeroShotSceneClassifier:
    """
    零样本场景分类器
    通过计算图像特征和场景文本描述的相似度进行分类
    """
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: BLIP2 模型包装器
            device: 设备
        """
        self.model = model
        self.device = device
        
        # 生成场景描述文本
        self.scene_prompts = self._generate_scene_prompts()
        
        print(f"  🔄 预计算场景文本特征...")
        
        # 创建 dummy 图像
        dummy_images = [Image.new('RGB', (224, 224), color=(128, 128, 128))] * len(self.scene_prompts)
        
        try:
            # 提取场景文本特征
            _, text_feats = model.extract_features(dummy_images, self.scene_prompts)
            
            if text_feats is None:
                raise ValueError("模型不支持文本特征提取")
            
            # 归一化文本特征
            self.scene_text_feats = F.normalize(text_feats, dim=-1)
            print(f"  ✅ 场景文本特征形状: {self.scene_text_feats.shape}")
        
        except Exception as e:
            print(f"  ❌ 文本特征提取失败: {e}")
            raise
    
    def _generate_scene_prompts(self):
        """生成场景描述文本"""
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
        print(f"  📝 场景描述示例: {prompts[0]}")
        return prompts
    
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
        
        # 归一化图像特征
        image_feats = F.normalize(image_feats, dim=-1)
        
        # 计算相似度作为 logits
        logits = image_feats @ self.scene_text_feats.t()
        
        # 缩放到更合理的范围
        logits = logits * 100.0
        
        return logits


# ============================================
# 扩展的模型包装器（支持零样本分类）
# ============================================
class ExtendedBLIP2Wrapper:
    """
    扩展的 BLIP2 模型包装器
    为没有场景分类头的模型添加零样本分类能力
    """
    def __init__(self, base_model, use_zero_shot=False):
        """
        Args:
            base_model: 原始 BLIP2 模型
            use_zero_shot: 是否使用零样本分类
        """
        self.base_model = base_model
        self.use_zero_shot = use_zero_shot
        self.zero_shot_classifier = None
        
        if use_zero_shot:
            print(f"  🎯 初始化零样本分类器...")
            try:
                self.zero_shot_classifier = ZeroShotSceneClassifier(
                    base_model,
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
        except Exception:
            pass
        
        # 使用零样本分类
        if self.use_zero_shot and self.zero_shot_classifier:
            return self.zero_shot_classifier.predict(images)
        
        return None
    
    @property
    def model_name(self):
        return getattr(self.base_model, 'model_name', 'BLIP2-LoRA')
    
    @property
    def device(self):
        return self.base_model.device


# ============================================
# 测试数据集
# ============================================
class ClassificationTestDataset(torch.utils.data.Dataset):
    """场景分类测试数据集"""
    
    def __init__(self, annotation_file: str, image_dir: str):
        print(f"📂 加载测试数据: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        
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
        print(f"✅ 加载 {len(self.data)} 个有效样本\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = os.path.join(self.image_dir, item['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 加载失败: {image_path}, {e}")
            image = Image.new('RGB', (224, 224), color='red')
        
        return {
            'image': image,
            'scene_label': item['scene_id'],
            'file_name': item['file_name'],
        }


# ============================================
# 评估单个模型
# ============================================
@torch.no_grad()
def evaluate_model(model, test_loader, device, exp_name):
    """
    评估单个模型的场景分类性能
    
    Args:
        model: 模型包装器
        test_loader: 测试数据加载器
        device: 设备
        exp_name: 实验名称
    
    Returns:
        results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"🔬 评估实验: {exp_name}")
    print(f"{'='*60}")
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    # 逐批次预测
    for batch in tqdm(test_loader, desc="推理中"):
        images = batch['image']
        labels = batch['scene_label']
        
        try:
            # 场景分类
            logits = model.classify_scene(images)
            
            if logits is None:
                raise ValueError(f"模型不支持场景分类")
            
            # 计算预测
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_labels.append(labels)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
        
        except Exception as e:
            print(f"⚠️ 批次处理失败: {e}")
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
    
    # 2. 每类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # 3. 加权平均
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 4. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 5. Top-3 准确率
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3_correct = np.array([label in top3_preds[i] for i, label in enumerate(all_labels)])
    top3_acc = top3_correct.mean() * 100
    
    # ========== 打印结果 ==========
    print(f"\n📊 分类结果:")
    print(f"  整体准确率: {accuracy:.2f}%")
    print(f"  Top-3 准确率: {top3_acc:.2f}%")
    print(f"  加权 F1 分数: {f1_avg*100:.2f}%")
    
    print(f"\n📋 各类别准确率:")
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
            print(f"  {scene_name:8s}: {class_acc:6.2f}%")
    
    # ========== 返回结果 ==========
    results = {
        'experiment': exp_name,
        'overall_accuracy': float(accuracy),
        'top3_accuracy': float(top3_acc),
        'weighted_f1': float(f1_avg * 100),
        'weighted_precision': float(precision_avg * 100),
        'weighted_recall': float(recall_avg * 100),
        'per_class': class_metrics,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels),
    }
    
    return results


# ============================================
# 可视化函数
# ============================================
def plot_comparison(all_results, output_dir):
    """绘制对比图"""
    
    # 提取数据
    exp_names = [r['experiment'] for r in all_results]
    overall_accs = [r['overall_accuracy'] for r in all_results]
    top3_accs = [r['top3_accuracy'] for r in all_results]
    f1_scores = [r['weighted_f1'] for r in all_results]
    
    # 中文实验名映射
    name_map = {
        'blip2_base': 'BLIP2 Base(零样本)',
        'itc_only': '仅ITC(零样本)',
        'scene_only': '仅场景',
        'itc_scene_equal': 'ITC+场景(1:1)',
        'itc_scene_2_8': 'ITC+场景(2:8)',
        'itc_scene_8_2': 'ITC+场景(8:2)',
        'itc_itm': 'ITC+ITM(零样本)',
        'itc_scene_itm': 'ITC+场景+ITM',
    }
    display_names = [name_map.get(name, name) for name in exp_names]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 整体准确率
    ax = axes[0, 0]
    bars = ax.barh(display_names, overall_accs, color='skyblue')
    ax.set_xlabel('整体准确率 (%)', fontsize=12)
    ax.set_title('场景分类准确率对比', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    for bar, acc in zip(bars, overall_accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)
    
    # 2. Top-3 准确率
    ax = axes[0, 1]
    bars = ax.barh(display_names, top3_accs, color='lightgreen')
    ax.set_xlabel('Top-3 准确率 (%)', fontsize=12)
    ax.set_title('Top-3 准确率对比', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    for bar, acc in zip(bars, top3_accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)
    
    # 3. F1 分数
    ax = axes[1, 0]
    bars = ax.barh(display_names, f1_scores, color='lightcoral')
    ax.set_xlabel('加权 F1 分数 (%)', fontsize=12)
    ax.set_title('F1 分数对比', fontsize=14, fontweight='bold')
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
    
    bars = ax.barh(display_names, avg_class_accs, color='plum')
    ax.set_xlabel('平均类别准确率 (%)', fontsize=12)
    ax.set_title('平均类别准确率对比', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    for bar, acc in zip(bars, avg_class_accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_scene_acc_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图已保存")
    plt.close()


def plot_per_class_heatmap(all_results, output_dir):
    """绘制各类别准确率热力图"""
    
    # 提取数据
    exp_names = [r['experiment'] for r in all_results]
    
    # 中文实验名映射
    name_map = {
        'blip2_base': 'BLIP2 Base(零样本)',
        'itc_only': '仅ITC(零样本)',
        'scene_only': '仅场景',
        'itc_scene_equal': 'ITC+场景(1:1)',
        'itc_scene_2_8': 'ITC+场景(2:8)',
        'itc_scene_8_2': 'ITC+场景(8:2)',
        'itc_itm': 'ITC+ITM(零样本)',
        'itc_scene_itm': 'ITC+场景+ITM',
    }
    display_names = [name_map.get(name, name) for name in exp_names]
    
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
    plt.figure(figsize=(14, max(6, len(exp_names) * 0.6)))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    sns.heatmap(
        data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=SCENE_CATEGORIES,
        yticklabels=display_names,
        cbar_kws={'label': '准确率 (%)'},
        vmin=0,
        vmax=100,
        linewidths=0.5,
    )
    
    plt.xlabel('场景类别', fontsize=12)
    plt.ylabel('消融实验', fontsize=12)
    plt.title('各场景类别准确率热力图', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'ablation_per_class_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    print(f"📊 热力图已保存")
    plt.close()


def save_comparison_table(all_results, output_dir):
    """保存对比表格"""
    
    # 总体指标表
    rows = []
    for result in all_results:
        row = {
            '实验名称': result['experiment'],
            '整体准确率(%)': f"{result['overall_accuracy']:.2f}",
            'Top-3准确率(%)': f"{result['top3_accuracy']:.2f}",
            'F1分数(%)': f"{result['weighted_f1']:.2f}",
            '精确率(%)': f"{result['weighted_precision']:.2f}",
            '召回率(%)': f"{result['weighted_recall']:.2f}",
            '样本数': result['num_samples'],
        }
        rows.append(row)
    
    df_overall = pd.DataFrame(rows)
    df_overall = df_overall.sort_values('整体准确率(%)', ascending=False)
    
    overall_file = os.path.join(output_dir, 'ablation_overall_comparison.csv')
    df_overall.to_csv(overall_file, index=False, encoding='utf-8-sig')
    print(f"\n📄 总体对比表格已保存: {overall_file}")
    
    # 打印表格
    print(f"\n{'='*80}")
    print(f"📊 消融实验场景分类准确率对比")
    print(f"{'='*80}")
    print(df_overall.to_string(index=False))
    print(f"{'='*80}")
    
    # 各类别详细表
    class_rows = []
    for result in all_results:
        for scene, metrics in result['per_class'].items():
            class_rows.append({
                '实验名称': result['experiment'],
                '场景类别': scene,
                '准确率(%)': f"{metrics['accuracy']:.2f}",
                '精确率(%)': f"{metrics['precision']:.2f}",
                '召回率(%)': f"{metrics['recall']:.2f}",
                'F1分数(%)': f"{metrics['f1']:.2f}",
                '样本数': metrics['support'],
            })
    
    df_class = pd.DataFrame(class_rows)
    class_file = os.path.join(output_dir, 'ablation_per_class_comparison.csv')
    df_class.to_csv(class_file, index=False, encoding='utf-8-sig')
    print(f"📄 类别详细对比表格已保存: {class_file}")
    
    return df_overall


# ============================================
# 主函数
# ============================================
def main():
    """主函数"""
    
    config = {
        # 数据
        'test_file': '/workspace/vlm/lab/output/test_split.json',
        'image_dir': '/data/fasion/train/image',
        'batch_size': 32,
        'num_workers': 4,
        
        # 设备
        'device': 'cuda:4',
        
        # 输出
        'output_dir': '/workspace/vlm/lab/output/ablation_scene_comparison',
        
        # 基础 BLIP2 权重
        'base_checkpoint': 'checkpoint_04.pth',
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("="*60)
    print("🔬 消融实验场景分类准确率对比")
    print("="*60)
    
    # ========== 定义要对比的消融实验 ==========
    ablation_experiments = [
        # 🔑 添加 BLIP2 Base 基线
        {
            'name': 'blip2_base',
            'display_name': 'BLIP2 Base(未微调零样本)',
            'lora_path': None,  # 不加载 LoRA
            'scene_head_path': None,
            'use_zero_shot': True,
            'is_base_model': True,  # 标记为基础模型
        },
        {
            'name': 'itc_only',
            'display_name': '仅ITC损失(零样本)',
            'lora_path': 'outputs/ablation_itc_only/best_model',
            'scene_head_path': None,
            'use_zero_shot': True,
            'is_base_model': False,
        },
        {
            'name': 'scene_only',
            'display_name': '仅场景分类损失',
            'lora_path': 'outputs/ablation_scene_only/best_model',
            'scene_head_path': 'outputs/ablation_scene_only/best_model/scene_head.pth',
            'use_zero_shot': False,
            'is_base_model': False,
        },
        {
            'name': 'itc_scene_equal',
            'display_name': 'ITC+场景(权重1:1)',
            'lora_path': 'outputs/ablation_itc_scene_equal/best_model',
            'scene_head_path': 'outputs/ablation_itc_scene_equal/best_model/scene_head.pth',
            'use_zero_shot': False,
            'is_base_model': False,
        },
        {
            'name': 'itc_scene_2_8',
            'display_name': 'ITC+场景(权重2:8)',
            'lora_path': 'outputs/ablation_itc_scene_2_8/best_model',
            'scene_head_path': 'outputs/ablation_itc_scene_2_8/best_model/scene_head.pth',
            'use_zero_shot': False,
            'is_base_model': False,
        },
        {
            'name': 'itc_scene_8_2',
            'display_name': 'ITC+场景(权重8:2)',
            'lora_path': 'outputs/ablation_itc_scene_8_2/best_model',
            'scene_head_path': 'outputs/ablation_itc_scene_8_2/best_model/scene_head.pth',
            'use_zero_shot': False,
            'is_base_model': False,
        },
        {
            'name': 'itc_itm',
            'display_name': 'ITC+ITM(零样本)',
            'lora_path': 'outputs/ablation_itc_itm/best_model',
            'scene_head_path': None,
            'use_zero_shot': True,
            'is_base_model': False,
        },
        {
            'name': 'itc_scene_itm',
            'display_name': 'ITC+场景+ITM(完整)',
            'lora_path': 'outputs/fashion_lora_itc_scene/best_model',
            'scene_head_path': 'outputs/fashion_lora_itc_scene/best_model/scene_head.pth',
            'use_zero_shot': False,
            'is_base_model': False,
        },
    ]
    
    # ========== 加载测试数据 ==========
    print(f"\n📂 加载测试数据...")
    test_dataset = ClassificationTestDataset(
        config['test_file'],
        config['image_dir']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=lambda x: {
            'image': [item['image'] for item in x],
            'scene_label': torch.tensor([item['scene_label'] for item in x]),
        }
    )
    
    # ========== 评估所有实验 ==========
    all_results = []
    
    for exp_config in ablation_experiments:
        try:
            print(f"\n{'#'*60}")
            print(f"# {exp_config['display_name']}")
            print(f"{'#'*60}")
            
            # 🔑 处理 BLIP2 Base 模型
            if exp_config.get('is_base_model', False):
                print(f"  📦 加载 BLIP2 Base 未微调模型...")
                
                # 直接加载基础 BLIP2 模型（不加载 LoRA）
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from models.blip2_qformer import Blip2Qformer
                
                base_model = Blip2Qformer(
                    vit_model="clip_L",
                    img_size=224,
                    freeze_vit=True,
                    num_query_token=32,
                    embed_dim=256,
                    max_txt_len=77,
                )
                
                # 加载基础权重
                checkpoint = torch.load(config['base_checkpoint'], map_location='cpu')
                state_dict = checkpoint.get("model", checkpoint)
                base_model.load_state_dict(state_dict, strict=False)
                base_model.to(config['device'])
                base_model.eval()
                
                # 设置图像预处理
                from torchvision import transforms
                normalize = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
                base_model.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ])
                
                # 添加必要的方法
                class BLIP2BaseWrapper:
                    def __init__(self, model, device):
                        self.model = model
                        self.device = device
                        self.model_name = "BLIP2-Base"
                    
                    @torch.no_grad()
                    def extract_features(self, images, texts):
                        if isinstance(images, list):
                            image_tensors = torch.stack([self.model.transform(img) for img in images])
                        else:
                            image_tensors = images
                        image_tensors = image_tensors.to(self.device)
                        
                        image_feats, text_feats = self.model({'image': image_tensors, 'text': texts})
                        
                        if image_feats.dim() == 3:
                            image_feats = image_feats.mean(dim=1)
                        
                        return image_feats, text_feats
                    
                    def classify_scene(self, images):
                        return None  # 没有分类头
                
                base_model = BLIP2BaseWrapper(base_model, config['device'])
                print(f"  ✅ BLIP2 Base 模型加载成功")
            
            else:
                # 处理微调模型
                if not os.path.exists(exp_config['lora_path']):
                    print(f"⚠️ LoRA 权重不存在: {exp_config['lora_path']}")
                    print(f"跳过...")
                    continue
                
                # 创建基础模型
                base_model = create_model(
                    'blip2-lora',
                    device=config['device'],
                    base_checkpoint=config['base_checkpoint'],
                    lora_checkpoint=exp_config['lora_path'],
                    scene_head_path=exp_config['scene_head_path'],
                )
            
            # 🔑 如果需要零样本分类，包装模型
            if exp_config.get('use_zero_shot', False):
                print(f"  🎯 使用零样本分类模式")
                model = ExtendedBLIP2Wrapper(base_model, use_zero_shot=True)
            else:
                model = base_model
            
            # 评估
            results = evaluate_model(
                model,
                test_loader,
                config['device'],
                exp_config['name']
            )
            
            if results is not None:
                results['display_name'] = exp_config['display_name']
                results['classification_method'] = 'zero-shot' if exp_config.get('use_zero_shot') else 'supervised'
                results['is_base_model'] = exp_config.get('is_base_model', False)
                all_results.append(results)
                
                # 保存单个实验结果
                result_file = os.path.join(config['output_dir'], 
                                          f"{exp_config['name']}_results.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 释放内存
            del model
            if 'base_model' in locals():
                del base_model
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"❌ 评估 {exp_config['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== 生成对比报告 ==========
    if len(all_results) == 0:
        print("\n❌ 没有成功评估的实验")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 生成对比报告...")
    print(f"{'='*60}")
    
    # 1. 保存汇总 JSON
    summary_file = os.path.join(config['output_dir'], 'ablation_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✅ 汇总结果已保存: {summary_file}")
    
    # 2. 保存对比表格
    df_overall = save_comparison_table(all_results, config['output_dir'])
    
    # 3. 绘制对比图
    plot_comparison(all_results, config['output_dir'])
    
    # 4. 绘制热力图
    plot_per_class_heatmap(all_results, config['output_dir'])
    
    # ========== 最终总结 ==========
    print(f"\n{'='*60}")
    print(f"✅ 消融实验对比完成！")
    print(f"{'='*60}")
    
    print(f"\n🏆 排名 (按场景分类准确率):")
    sorted_results = sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        display_name = result.get('display_name', result['experiment'])
        method = result.get('classification_method', 'unknown')
        is_base = result.get('is_base_model', False)
        
        tags = []
        if is_base:
            tags.append('baseline')
        if method == 'zero-shot':
            tags.append('zero-shot')
        tag_str = f"[{', '.join(tags)}]" if tags else ""
        
        print(f"  {i}. {display_name:<35s} {tag_str:<20s}: {result['overall_accuracy']:.2f}%")
    
    print(f"\n💡 最佳配置: {sorted_results[0].get('display_name', sorted_results[0]['experiment'])}")
    print(f"   准确率: {sorted_results[0]['overall_accuracy']:.2f}%")
    print(f"   方法: {sorted_results[0].get('classification_method', 'supervised')}")
    
    # 对比基线的提升
    base_result = next((r for r in all_results if r.get('is_base_model')), None)
    best_result = sorted_results[0]
    if base_result and not best_result.get('is_base_model'):
        improvement = best_result['overall_accuracy'] - base_result['overall_accuracy']
        print(f"\n📈 相比 BLIP2 Base 提升: +{improvement:.2f}%")
    
    print(f"\n📁 所有结果已保存到: {config['output_dir']}")
    print(f"\n📊 查看可视化结果:")
    print(f"   对比图: {config['output_dir']}/ablation_scene_acc_comparison.png")
    print(f"   热力图: {config['output_dir']}/ablation_per_class_heatmap.png")


if __name__ == "__main__":
    main()