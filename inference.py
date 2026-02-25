#!/usr/bin/env python3
"""
BLIP2 + LoRA 图文检索推理脚本
用于实际应用场景的图文检索
支持 text + scene_category 拼接格式
"""

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from peft import PeftModel
from models.blip2_qformer import Blip2Qformer


# ============================================
# 场景类别定义
# ============================================
SCENE_CATEGORIES = [
    '职场正装', '职场休闲', '运动健身', '户外探险', '居家休闲',
    '社交聚会', '旅行度假', '运动赛事', '婚礼相关', '特殊功能',
]


class LoRARetrievalPipeline:
    """带 LoRA 的图文检索 Pipeline"""
    
    def __init__(self, base_checkpoint, lora_checkpoint, device='cuda:4', use_scene_suffix=True):
        """
        Args:
            base_checkpoint: 基础 BLIP2 权重路径
            lora_checkpoint: LoRA 适配器目录
            device: 设备
            use_scene_suffix: 是否在文本后添加场景类别后缀
        """
        self.device = device
        self.use_scene_suffix = use_scene_suffix
        self.model = self._load_model(base_checkpoint, lora_checkpoint)
        self.transform = self._get_transform()
        
        if use_scene_suffix:
            print(f"💡 使用场景后缀格式: text + \"，适合\" + scene_category")
        else:
            print(f"💡 使用原始文本格式")
    
    def _load_model(self, base_checkpoint, lora_checkpoint):
        """加载模型"""
        print(f"📥 加载模型...")
        
        # 基础模型
        model = Blip2Qformer(
            vit_model="clip_L",
            img_size=224,
            freeze_vit=True,
            num_query_token=32,
            embed_dim=256,
            max_txt_len=77,
        )
        
        # 加载基础权重
        if os.path.exists(base_checkpoint):
            checkpoint = torch.load(base_checkpoint, map_location='cpu')
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ 基础权重已加载")
        
        # 加载 LoRA
        if os.path.exists(lora_checkpoint):
            model.Qformer = PeftModel.from_pretrained(
                model.Qformer,
                lora_checkpoint,
                is_trainable=False
            )
            print(f"✅ LoRA 权重已加载")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self):
        """图像预处理"""
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    
    def format_text_with_scene(self, text, scene_category=None):
        """
        格式化文本（添加场景后缀）
        
        Args:
            text: 原始文本描述
            scene_category: 场景类别（可选）
        
        Returns:
            formatted_text: 格式化后的文本
        """
        if not self.use_scene_suffix:
            return text
        
        if scene_category:
            return f"{text}，适合{scene_category}"
        else:
            return text
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        return image
    
    @torch.no_grad()
    def compute_similarity(self, images, texts):
        """
        计算图文相似度
        
        Args:
            images: 图像路径列表或图像 tensor
            texts: 文本列表（或包含 text/scene_category 的字典列表）
        
        Returns:
            similarity_matrix: [N_images, N_texts] 相似度矩阵
        """
        # 处理图像
        if isinstance(images, list):
            image_tensors = []
            for img_path in images:
                img = self.preprocess_image(img_path)
                image_tensors.append(img)
            images = torch.cat(image_tensors, dim=0)
        
        images = images.to(self.device)
        
        # 处理文本（支持字典格式）
        formatted_texts = []
        for text in texts:
            if isinstance(text, dict):
                # 字典格式: {'text': ..., 'scene_category': ...}
                formatted_text = self.format_text_with_scene(
                    text.get('text', ''),
                    text.get('scene_category', None)
                )
            else:
                # 字符串格式
                formatted_text = text
            formatted_texts.append(formatted_text)
        
        # 提取特征
        image_feats, text_feats = self.model({'image': images, 'text': formatted_texts})
        
        # 处理维度
        if image_feats.dim() == 3:
            image_feats = image_feats.mean(dim=1)
        
        # 归一化
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        
        # 计算相似度
        similarity = image_feats @ text_feats.t()
        
        return similarity.cpu()
    
    def retrieve_text(self, image_path, text_candidates, top_k=5):
        """
        给定图像，检索最相关的文本
        
        Args:
            image_path: 图像路径
            text_candidates: 候选文本列表（字符串或字典）
            top_k: 返回前 k 个结果
        
        Returns:
            results: [(text, score), ...]
        """
        similarity = self.compute_similarity([image_path], text_candidates)
        similarity = similarity[0]  # [N_texts]
        
        # 排序
        scores, indices = torch.topk(similarity, k=min(top_k, len(text_candidates)))
        
        results = []
        for score, idx in zip(scores, indices):
            text_item = text_candidates[idx.item()]
            
            # 返回原始文本或字典
            if isinstance(text_item, dict):
                display_text = self.format_text_with_scene(
                    text_item.get('text', ''),
                    text_item.get('scene_category', None)
                )
                results.append((display_text, score.item(), text_item))
            else:
                results.append((text_item, score.item()))
        
        return results
    
    def retrieve_image(self, text, image_paths, top_k=5, scene_category=None):
        """
        给定文本，检索最相关的图像
        
        Args:
            text: 查询文本（字符串或字典）
            image_paths: 候选图像路径列表
            top_k: 返回前 k 个结果
            scene_category: 场景类别（可选，如果 text 是字符串时使用）
        
        Returns:
            results: [(image_path, score), ...]
        """
        # 处理输入文本
        if isinstance(text, dict):
            query_texts = [text]
        else:
            query_texts = [{'text': text, 'scene_category': scene_category}]
        
        similarity = self.compute_similarity(image_paths, query_texts)
        similarity = similarity[:, 0]  # [N_images]
        
        # 排序
        scores, indices = torch.topk(similarity, k=min(top_k, len(image_paths)))
        
        results = []
        for score, idx in zip(scores, indices):
            results.append((image_paths[idx.item()], score.item()))
        
        return results
    
    def retrieve_from_annotation(self, query_text, annotation_file, image_dir, 
                                  top_k=5, scene_category=None):
        """
        从标注文件中检索图像
        
        Args:
            query_text: 查询文本
            annotation_file: 标注文件路径
            image_dir: 图像目录
            top_k: 返回前 k 个结果
            scene_category: 场景类别（可选）
        
        Returns:
            results: [(image_path, score, annotation), ...]
        """
        # 加载标注
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 构建图像路径
        image_paths = [os.path.join(image_dir, ann['file_name']) for ann in annotations]
        
        # 检索
        results = self.retrieve_image(query_text, image_paths, top_k, scene_category)
        
        # 添加标注信息
        results_with_ann = []
        for img_path, score in results:
            idx = image_paths.index(img_path)
            results_with_ann.append((img_path, score, annotations[idx]))
        
        return results_with_ann


# ============================================
# 使用示例
# ============================================
def demo_basic():
    """基础推理示例"""
    
    print("\n" + "="*60)
    print("📝 基础推理示例")
    print("="*60)
    
    # 初始化 Pipeline
    pipeline = LoRARetrievalPipeline(
        base_checkpoint='checkpoint_04.pth',
        lora_checkpoint='outputs/fashion_lora_itc_scene/best_model',
        device='cuda:4',
        use_scene_suffix=True  # 使用场景后缀
    )
    
    # 示例1: 图像检索文本（使用字典格式）
    print("\n" + "-"*60)
    print("示例1: 图像检索文本（带场景类别）")
    print("-"*60)
    
    image_path = "/data/fasion/train/image/010207.jpg"
    
    # 候选文本（字典格式）
    text_candidates = [
        {'text': '修身剪裁西装套装，深色商务风格', 'scene_category': '职场正装'},
        {'text': '轻便透气运动服，适合日常锻炼', 'scene_category': '运动健身'},
        {'text': '柔软舒适家居服，休闲宽松', 'scene_category': '居家休闲'},
        {'text': '多功能户外冲锋衣，防风防水', 'scene_category': '户外探险'},
        {'text': '优雅晚礼服，华丽设计', 'scene_category': '社交聚会'},
    ]
    
    results = pipeline.retrieve_text(image_path, text_candidates, top_k=3)
    
    print(f"查询图像: {os.path.basename(image_path)}")
    print(f"最匹配的文本:")
    for i, result in enumerate(results, 1):
        if len(result) == 3:  # (text, score, dict)
            text, score, orig = result
            print(f"  {i}. {text}")
            print(f"     相似度: {score:.4f}")
        else:  # (text, score)
            text, score = result
            print(f"  {i}. {text} (相似度: {score:.4f})")
    
    # 示例2: 文本检索图像（带场景类别）
    print("\n" + "-"*60)
    print("示例2: 文本检索图像（带场景类别）")
    print("-"*60)
    
    query_text = "修身商务西装，适合正式场合"
    scene_category = "职场正装"
    
    image_paths = [
        "/data/fasion/train/image/010207.jpg",
        "/data/fasion/train/image/010208.jpg",
        "/data/fasion/train/image/010209.jpg",
    ]
    
    results = pipeline.retrieve_image(
        query_text, 
        image_paths, 
        top_k=3,
        scene_category=scene_category
    )
    
    print(f"查询文本: {query_text}，适合{scene_category}")
    print(f"最匹配的图像:")
    for i, (img_path, score) in enumerate(results, 1):
        print(f"  {i}. {os.path.basename(img_path)} (相似度: {score:.4f})")


def demo_with_annotation():
    """使用标注文件的推理示例"""
    
    print("\n" + "="*60)
    print("📚 基于标注文件的检索示例")
    print("="*60)
    
    # 初始化 Pipeline
    pipeline = LoRARetrievalPipeline(
        base_checkpoint='checkpoint_04.pth',
        lora_checkpoint='outputs/fashion_lora_itc_scene/best_model',
        device='cuda:4',
        use_scene_suffix=True
    )
    
    # 从标注文件检索
    query_text = "适合夏季穿着的轻薄衣物"
    scene_category = "旅行度假"
    
    results = pipeline.retrieve_from_annotation(
        query_text=query_text,
        annotation_file='/workspace/vlm/lab/output/test_split.json',
        image_dir='/data/fasion/train/image',
        top_k=5,
        scene_category=scene_category
    )
    
    print(f"\n查询: {query_text}，适合{scene_category}")
    print(f"\n检索结果:")
    for i, (img_path, score, ann) in enumerate(results, 1):
        print(f"\n{i}. {os.path.basename(img_path)} (相似度: {score:.4f})")
        print(f"   场景: {ann.get('scene_category', 'N/A')}")
        print(f"   描述: {ann.get('text', 'N/A')[:80]}...")


def demo_batch_retrieval():
    """批量检索示例"""
    
    print("\n" + "="*60)
    print("🔄 批量检索示例")
    print("="*60)
    
    pipeline = LoRARetrievalPipeline(
        base_checkpoint='checkpoint_04.pth',
        lora_checkpoint='outputs/fashion_lora_itc_scene/best_model',
        device='cuda:4',
        use_scene_suffix=True
    )
    
    # 多个查询
    queries = [
        {'text': '商务正装西装', 'scene_category': '职场正装'},
        {'text': '休闲运动装备', 'scene_category': '运动健身'},
        {'text': '舒适家居服', 'scene_category': '居家休闲'},
    ]
    
    image_paths = [
        "/data/fasion/train/image/010207.jpg",
        "/data/fasion/train/image/010208.jpg",
        "/data/fasion/train/image/010209.jpg",
    ]
    
    # 计算所有相似度
    similarity = pipeline.compute_similarity(image_paths, queries)
    
    print(f"\n相似度矩阵 [{len(image_paths)} 图像 × {len(queries)} 查询]:")
    print("-" * 60)
    print(f"{'图像':<20s}", end="")
    for i, q in enumerate(queries):
        print(f"查询{i+1:<3d}", end="  ")
    print()
    print("-" * 60)
    
    for i, img_path in enumerate(image_paths):
        print(f"{os.path.basename(img_path):<20s}", end="")
        for j in range(len(queries)):
            print(f"{similarity[i, j]:.4f}  ", end="")
        print()


if __name__ == "__main__":
    # 运行所有示例
    demo_basic()
    #demo_with_annotation()
    demo_batch_retrieval()