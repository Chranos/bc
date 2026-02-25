"""
模型库 - 统一接口封装不同的预训练模型
支持的模型:
- CLIP (OpenAI): ViT-B/32, ViT-L/14
- Chinese-CLIP: 中文多模态模型
- BLIP: Salesforce 图文检索模型
- BLIP2 + LoRA: 微调模型（你的模型）
- ResNet: 纯图像分类基线
- ViT: Vision Transformer 分类基线
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


# ============================================
# 基础模型包装器
# ============================================
class BaseModelWrapper(ABC):
    """基础模型包装器 - 统一接口"""
    
    def __init__(self, device='cuda'):
        """
        Args:
            device: 运行设备 (cuda/cpu)
        """
        self.device = device
        self.model = None
        self.processor = None
        self.model_name = "BaseModel"
    
    @abstractmethod
    def extract_features(self, images: Union[List[Image.Image], torch.Tensor], 
                        texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        提取图像和文本特征
        
        Args:
            images: PIL Images 列表或 tensor [B, C, H, W]
            texts: 文本列表
        
        Returns:
            image_feats: [B, D] 图像特征
            text_feats: [B, D] 文本特征（如果支持）或 None
        """
        pass
    
    @abstractmethod
    def classify_scene(self, images: Union[List[Image.Image], torch.Tensor]) -> Optional[torch.Tensor]:
        """
        场景分类（如果支持）
        
        Args:
            images: PIL Images 列表或 tensor
        
        Returns:
            logits: [B, num_classes] 分类 logits 或 None
        """
        pass
    
    def compute_similarity(self, image_feats: torch.Tensor, 
                          text_feats: torch.Tensor) -> torch.Tensor:
        """
        计算图文相似度矩阵
        
        Args:
            image_feats: [N, D]
            text_feats: [M, D]
        
        Returns:
            similarity: [N, M]
        """
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        return image_feats @ text_feats.t()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'name': self.model_name,
            'device': str(self.device),
            'supports_text': True,
            'supports_classification': False,
        }


# ============================================
# CLIP 模型封装
# ============================================
class CLIPWrapper(BaseModelWrapper):
    """OpenAI CLIP 模型"""
    
    def __init__(self, model_name='ViT-B/32', device='cuda', checkpoint_path=None, num_classes=10):
        """
        Args:
            model_name: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'
            device: 运行设备
            checkpoint_path: 微调后的权重路径（可选）
            num_classes: 分类类别数
        """
        super().__init__(device)
        self.model_name = f"CLIP-{model_name}"
        self.clip_model_name = model_name
        
        print(f"📥 加载 {self.model_name}...")
        
        try:
            import clip
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            print(f"✅ {self.model_name} 加载成功")
        except Exception as e:
            raise RuntimeError(f"❌ 加载 CLIP 失败: {e}\n请安装: pip install git+https://github.com/openai/CLIP.git")
        
        # 获取特征维度
        if 'ViT-B' in model_name:
            self.feature_dim = 512
        elif 'ViT-L' in model_name:
            self.feature_dim = 768
        elif 'RN50' in model_name or 'RN101' in model_name:
            self.feature_dim = 1024
        else:
            self.feature_dim = 512
        
        # 分类器（如果提供了checkpoint则加载微调权重）
        self.classifier = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📥 加载微调权重: {checkpoint_path}")
            
            # 重建分类器结构（必须和训练时一致！）
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            ).to(device)
            
            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 提取 classifier 权重
            classifier_state = {}
            clip_state = {}
            
            for key, value in state_dict.items():
                if 'classifier.' in key:
                    new_key = key.replace('classifier.', '')
                    classifier_state[new_key] = value
                elif 'clip_model.' in key:
                    new_key = key.replace('clip_model.', '')
                    clip_state[new_key] = value
            
            # 加载分类器
            if classifier_state:
                try:
                    self.classifier.load_state_dict(classifier_state, strict=True)
                    print(f"  ✅ 分类器权重已加载 ({len(classifier_state)} 个参数)")
                except Exception as e:
                    print(f"  ⚠️ 分类器权重加载失败: {e}")
                    print(f"  Keys in checkpoint: {list(classifier_state.keys())[:5]}...")
            else:
                print(f"  ⚠️ Checkpoint 中没有 classifier 权重")
            
            # 加载 CLIP 编码器（可选，通常冻结不需要）
            if clip_state:
                try:
                    self.model.load_state_dict(clip_state, strict=False)
                    print(f"  ✅ CLIP 编码器权重已更新")
                except Exception as e:
                    print(f"  ⚠️ CLIP 权重加载失败（使用默认）: {e}")
            
            self.classifier.eval()
    
    @torch.no_grad()
    def extract_features(self, images, texts):
        """提取 CLIP 特征"""
        # 处理图像
        if isinstance(images, torch.Tensor):
            # 如果是 tensor，需要转换为 PIL
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            images = [to_pil(img.cpu()) for img in images]
        
        # 预处理图像
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        # 处理文本
        import clip
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        
        # 提取特征
        image_feats = self.model.encode_image(image_inputs)
        text_feats = self.model.encode_text(text_tokens)
        
        # 转换为 float32
        image_feats = image_feats.float()
        text_feats = text_feats.float()
        
        return image_feats, text_feats
    
    @torch.no_grad()
    def classify_scene(self, images):
        """场景分类"""
        if self.classifier is None:
            return None
        
        # 提取图像特征
        image_feats, _ = self.extract_features(images, [""])
        
        # 分类
        logits = self.classifier(image_feats)
        return logits
    
    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            'supports_text': True,
            'supports_classification': self.classifier is not None,
            'model_type': 'CLIP',
        })
        return info


# ============================================
# Chinese-CLIP 模型封装
# ============================================
class ChineseCLIPWrapper(BaseModelWrapper):
    """Chinese-CLIP 中文多模态模型"""
    
    def __init__(self, model_name='OFA-Sys/chinese-clip-vit-base-patch16', device='cuda'):
        """
        Args:
            model_name: HuggingFace 模型名称
            device: 运行设备
        """
        super().__init__(device)
        self.model_name = "Chinese-CLIP"
        
        print(f"📥 加载 {self.model_name}...")
        
        try:
            from transformers import ChineseCLIPProcessor, ChineseCLIPModel
            
            self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
            self.model = ChineseCLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
            
            print(f"✅ {self.model_name} 加载成功")
        except Exception as e:
            raise RuntimeError(f"❌ 加载 Chinese-CLIP 失败: {e}\n请安装: pip install transformers")
    
    @torch.no_grad()
    def extract_features(self, images, texts):
        """提取 Chinese-CLIP 特征"""
        # 转换图像格式
        if isinstance(images, torch.Tensor):
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            images = [to_pil(img.cpu()) for img in images]
        
        # 处理输入
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        outputs = self.model(**inputs)
        image_feats = outputs.image_embeds
        text_feats = outputs.text_embeds
        
        return image_feats, text_feats
    
    def classify_scene(self, images):
        """Chinese-CLIP 不直接支持分类"""
        return None
    
    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            'supports_text': True,
            'supports_classification': False,
            'model_type': 'Chinese-CLIP',
            'language': 'Chinese',
        })
        return info


# ============================================
# BLIP 模型封装
# ============================================
class BLIPWrapper(BaseModelWrapper):
    """BLIP 图文检索模型"""
    
    def __init__(self, model_name='Salesforce/blip-itm-base-coco', device='cuda'):
        """
        Args:
            model_name: HuggingFace 模型名称
            device: 运行设备
        """
        super().__init__(device)
        self.model_name = "BLIP-Base"
        
        print(f"📥 加载 {self.model_name}...")
        
        try:
            from transformers import BlipProcessor, BlipModel
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipModel.from_pretrained(model_name).to(device)
            self.model.eval()
            
            print(f"✅ {self.model_name} 加载成功")
        except Exception as e:
            raise RuntimeError(f"❌ 加载 BLIP 失败: {e}\n请安装: pip install transformers")
    
    @torch.no_grad()
    def extract_features(self, images, texts):
        """提取 BLIP 特征"""
        # 转换图像格式
        if isinstance(images, torch.Tensor):
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            images = [to_pil(img.cpu()) for img in images]
        
        # 处理输入
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        outputs = self.model(**inputs)
        
        # BLIP 的特征在 vision_model_output 和 text_model_output 中
        image_feats = outputs.image_embeds
        text_feats = outputs.text_embeds
        
        return image_feats, text_feats
    
    def classify_scene(self, images):
        """BLIP 不直接支持分类"""
        return None
    
    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            'supports_text': True,
            'supports_classification': False,
            'model_type': 'BLIP',
        })
        return info


# ============================================
# BLIP2 + LoRA 模型封装
# ============================================
class BLIP2LoRAWrapper(BaseModelWrapper):
    """BLIP2 + LoRA 微调模型（你的模型）"""
    
    def __init__(self, base_checkpoint: str, lora_checkpoint: str, 
                 scene_head_path: Optional[str] = None, device='cuda'):
        """
        Args:
            base_checkpoint: 基础 BLIP2 权重路径
            lora_checkpoint: LoRA 适配器目录
            scene_head_path: 场景分类头权重路径（可选）
            device: 运行设备
        """
        super().__init__(device)
        self.model_name = "BLIP2-LoRA (Ours)"
        
        print(f"📥 加载 {self.model_name}...")
        
        # 检查文件存在
        if not os.path.exists(base_checkpoint):
            raise FileNotFoundError(f"基础权重不存在: {base_checkpoint}")
        if not os.path.exists(lora_checkpoint):
            raise FileNotFoundError(f"LoRA 权重不存在: {lora_checkpoint}")
        
        try:
            # 导入本地模型
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from models.blip2_qformer import Blip2Qformer
            from peft import PeftModel
            
            # 加载基础模型
            self.model = Blip2Qformer(
                vit_model="clip_L",
                img_size=224,
                freeze_vit=True,
                num_query_token=32,
                embed_dim=256,
                max_txt_len=77,
            )
            
            # 加载基础权重
            checkpoint = torch.load(base_checkpoint, map_location='cpu')
            state_dict = checkpoint.get("model", checkpoint)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            
            if missing:
                print(f"  ⚠️ 缺失的键: {len(missing)} 个")
            if unexpected:
                print(f"  ⚠️ 未预期的键: {len(unexpected)} 个")
            
            # 加载 LoRA
            self.model.Qformer = PeftModel.from_pretrained(
                self.model.Qformer,
                lora_checkpoint,
                is_trainable=False
            )
            
            self.model.to(device)
            self.model.eval()
            
            print(f"✅ BLIP2 + LoRA 加载成功")
            
            # 加载场景分类头
            self.scene_head = None
            if scene_head_path and os.path.exists(scene_head_path):
                # 定义场景分类头结构（和训练时一致）
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
                
                self.scene_head = SceneClassificationHead(
                    input_dim=256,
                    num_classes=10,
                    dropout=0.1
                ).to(device)
                
                state_dict = torch.load(scene_head_path, map_location='cpu')
                self.scene_head.load_state_dict(state_dict)
                self.scene_head.eval()
                
                print(f"✅ 场景分类头加载成功")
            
            # 设置图像预处理
            from torchvision import transforms
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"❌ 加载 BLIP2-LoRA 失败: {e}")
    
    @torch.no_grad()
    def extract_features(self, images, texts):
        """提取 BLIP2 特征"""
        # 处理图像
        if isinstance(images, list):
            # PIL Images
            image_tensors = torch.stack([self.transform(img) for img in images])
        else:
            # 已经是 tensor
            image_tensors = images
        
        image_tensors = image_tensors.to(self.device)
        
        # 提取特征
        image_feats, text_feats = self.model({'image': image_tensors, 'text': texts})
        
        # 处理维度
        if image_feats.dim() == 3:
            # [B, num_queries, D] -> [B, D]
            image_feats = image_feats.mean(dim=1)
        
        return image_feats, text_feats
    
    @torch.no_grad()
    def classify_scene(self, images):
        """场景分类"""
        if self.scene_head is None:
            print("⚠️ 场景分类头未加载")
            return None
        
        # 提取图像特征
        image_feats, _ = self.extract_features(images, [""] * len(images))
        
        # 分类
        logits = self.scene_head(image_feats)
        return logits
    
    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            'supports_text': True,
            'supports_classification': self.scene_head is not None,
            'model_type': 'BLIP2-LoRA',
            'has_lora': True,
        })
        return info


# ============================================
# ResNet 分类基线
# ============================================
class ResNetClassifier(BaseModelWrapper):
    """ResNet 场景分类基线"""
    
    def __init__(self, num_classes=10, device='cuda', pretrained=True, 
                 checkpoint_path: Optional[str] = None):
        """
        Args:
            num_classes: 分类类别数
            device: 运行设备
            pretrained: 是否使用 ImageNet 预训练
            checkpoint_path: 微调后的权重路径（可选）
        """
        super().__init__(device)
        self.model_name = "ResNet-50"
        
        print(f"📥 加载 {self.model_name}...")
        
        try:
            from torchvision import models, transforms
            
            # 加载模型
            self.model = models.resnet50(pretrained=pretrained)
            
            # 获取特征维度
            feature_dim = self.model.fc.in_features
            
            # 替换分类层（和训练时一致！）
            self.model.fc = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
            # 加载微调权重
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"📥 加载微调权重: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 处理可能的 key 前缀
                model_state = {}
                for key, value in state_dict.items():
                    new_key = key.replace('backbone.', '') if 'backbone.' in key else key
                    model_state[new_key] = value
                
                try:
                    self.model.load_state_dict(model_state, strict=True)
                    print(f"  ✅ 微调权重已加载")
                except Exception as e:
                    print(f"  ⚠️ 权重加载失败，尝试部分加载: {e}")
                    self.model.load_state_dict(model_state, strict=False)
            
            self.model.to(device)
            self.model.eval()
            
            # 图像预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print(f"✅ {self.model_name} 加载成功")
            
        except Exception as e:
            raise RuntimeError(f"❌ 加载 ResNet 失败: {e}")
    
    @torch.no_grad()
    def extract_features(self, images, texts):
        """提取 ResNet 特征（不支持文本）"""
        # 处理图像
        if isinstance(images, list):
            image_tensors = torch.stack([self.transform(img) for img in images])
        else:
            image_tensors = images
        
        image_tensors = image_tensors.to(self.device)
        
        # 提取特征（去掉最后的分类层）
        x = self.model.conv1(image_tensors)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features, None  # 不支持文本特征
    
    @torch.no_grad()
    def classify_scene(self, images):
        """场景分类"""
        # 处理图像
        if isinstance(images, list):
            image_tensors = torch.stack([self.transform(img) for img in images])
        else:
            image_tensors = images
        
        image_tensors = image_tensors.to(self.device)
        
        # 分类
        logits = self.model(image_tensors)
        return logits
    
    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            'supports_text': False,
            'supports_classification': True,
            'model_type': 'ResNet',
        })
        return info


# ============================================
# ViT 分类基线
# ============================================
class ViTClassifier(BaseModelWrapper):
    """Vision Transformer 场景分类基线"""
    
    def __init__(self, model_name='vit_base_patch16_224', num_classes=10, 
                 device='cuda', pretrained=True, checkpoint_path: Optional[str] = None):
        """
        Args:
            model_name: ViT 模型名称
            num_classes: 分类类别数
            device: 运行设备
            pretrained: 是否使用预训练
            checkpoint_path: 微调后的权重路径（可选）
        """
        super().__init__(device)
        self.model_name = f"ViT-{model_name}"
        
        print(f"📥 加载 {self.model_name}...")
        
        try:
            import timm
            from torchvision import transforms
            
            # 加载 backbone（不带分类头）
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # 不要分类头
            )
            
            # 获取特征维度
            feature_dim = self.backbone.num_features
            
            # 创建分类头（和训练时一致！）
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
            # 加载微调权重
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"📥 加载微调权重: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 分离 backbone 和 classifier
                backbone_state = {}
                classifier_state = {}
                
                for key, value in state_dict.items():
                    if 'backbone.' in key:
                        new_key = key.replace('backbone.', '')
                        backbone_state[new_key] = value
                    elif 'classifier.' in key:
                        new_key = key.replace('classifier.', '')
                        classifier_state[new_key] = value
                
                # 加载权重
                if backbone_state:
                    self.backbone.load_state_dict(backbone_state, strict=False)
                    print(f"  ✅ Backbone 权重已加载")
                
                if classifier_state:
                    self.classifier.load_state_dict(classifier_state, strict=True)
                    print(f"  ✅ 分类器权重已加载")
            
            self.backbone.to(device)
            self.classifier.to(device)
            self.backbone.eval()
            self.classifier.eval()
            
            # 图像预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print(f"✅ {self.model_name} 加载成功")
            
        except Exception as e:
            raise RuntimeError(f"❌ 加载 ViT 失败: {e}\n请安装: pip install timm")
    
    @torch.no_grad()
    def extract_features(self, images, texts):
        """提取 ViT 特征（不支持文本）"""
        # 处理图像
        if isinstance(images, list):
            image_tensors = torch.stack([self.transform(img) for img in images])
        else:
            image_tensors = images
        
        image_tensors = image_tensors.to(self.device)
        
        # 提取特征
        features = self.backbone(image_tensors)
        
        # 处理维度
        if features.dim() == 3:  # [B, N, D]
            features = features.mean(dim=1)  # [B, D]
        
        return features, None
    
    @torch.no_grad()
    def classify_scene(self, images):
        """场景分类"""
        # 处理图像
        if isinstance(images, list):
            image_tensors = torch.stack([self.transform(img) for img in images])
        else:
            image_tensors = images
        
        image_tensors = image_tensors.to(self.device)
        
        # 提取特征 + 分类
        features = self.backbone(image_tensors)
        logits = self.classifier(features)
        return logits
    
    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            'supports_text': False,
            'supports_classification': True,
            'model_type': 'ViT',
        })
        return info


# ============================================
# 模型工厂
# ============================================
def create_model(model_name: str, device='cuda', **kwargs) -> BaseModelWrapper:
    """
    创建模型
    
    Args:
        model_name: 模型名称
            - 'clip-vit-b32': CLIP ViT-B/32
            - 'clip-vit-b16': CLIP ViT-B/16
            - 'clip-vit-l14': CLIP ViT-L/14
            - 'clip-rn50': CLIP ResNet-50
            - 'chinese-clip': Chinese-CLIP
            - 'blip-base': BLIP Base
            - 'blip2-lora': BLIP2 + LoRA (你的模型)
            - 'resnet50': ResNet-50
            - 'vit-base': ViT-Base
        device: 设备 (cuda/cpu)
        **kwargs: 额外参数
            - checkpoint_path: 微调权重路径
            - num_classes: 分类类别数 (default: 10)
            - base_checkpoint: BLIP2 基础权重 (for blip2-lora)
            - lora_checkpoint: LoRA 权重 (for blip2-lora)
            - scene_head_path: 分类头权重 (for blip2-lora)
    
    Returns:
        model: 模型包装器实例
    """
    
    model_registry = {
        # CLIP 系列
        'clip-vit-b32': lambda: CLIPWrapper('ViT-B/32', device, **kwargs),
        'clip-vit-b16': lambda: CLIPWrapper('ViT-B/16', device, **kwargs),
        'clip-vit-l14': lambda: CLIPWrapper('ViT-L/14', device, **kwargs),
        'clip-rn50': lambda: CLIPWrapper('RN50', device, **kwargs),
        'clip-rn101': lambda: CLIPWrapper('RN101', device, **kwargs),
        
        # Chinese-CLIP
        'chinese-clip': lambda: ChineseCLIPWrapper(device=device),
        'chinese-clip-large': lambda: ChineseCLIPWrapper(
            model_name='OFA-Sys/chinese-clip-vit-large-patch14',
            device=device
        ),
        
        # BLIP
        'blip-base': lambda: BLIPWrapper(device=device),
        'blip-large': lambda: BLIPWrapper(
            model_name='Salesforce/blip-itm-large-coco',
            device=device
        ),
        
        # BLIP2 + LoRA
        'blip2-lora': lambda: BLIP2LoRAWrapper(
            base_checkpoint=kwargs.get('base_checkpoint'),
            lora_checkpoint=kwargs.get('lora_checkpoint'),
            scene_head_path=kwargs.get('scene_head_path'),
            device=device
        ),
        
        # 分类基线
        'resnet50': lambda: ResNetClassifier(
            num_classes=kwargs.get('num_classes', 10),
            device=device,
            pretrained=kwargs.get('pretrained', True),
            checkpoint_path=kwargs.get('checkpoint_path'),
        ),
        'vit-base': lambda: ViTClassifier(
            model_name='vit_base_patch16_224',
            num_classes=kwargs.get('num_classes', 10),
            device=device,
            pretrained=kwargs.get('pretrained', True),
            checkpoint_path=kwargs.get('checkpoint_path'),
        ),
    }
    
    if model_name not in model_registry:
        available = ', '.join(model_registry.keys())
        raise ValueError(
            f"❌ 未知的模型: {model_name}\n"
            f"可用模型: {available}"
        )
    
    try:
        model = model_registry[model_name]()
        print(f"✅ 模型创建成功: {model.model_name}")
        return model
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"❌ 创建模型 {model_name} 失败: {e}")


# ============================================
# 辅助函数
# ============================================
def list_available_models() -> List[str]:
    """列出所有可用的模型"""
    return [
        'clip-vit-b32', 'clip-vit-b16', 'clip-vit-l14', 'clip-rn50',
        'chinese-clip', 'blip-base', 'blip2-lora', 
        'resnet50', 'vit-base'
    ]


def print_model_info(model: BaseModelWrapper):
    """打印模型信息"""
    info = model.get_model_info()
    print(f"\n{'='*60}")
    print(f"📋 模型信息: {info['name']}")
    print(f"{'='*60}")
    print(f"  设备: {info['device']}")
    print(f"  支持文本: {'✅' if info['supports_text'] else '❌'}")
    print(f"  支持分类: {'✅' if info['supports_classification'] else '❌'}")
    print(f"  模型类型: {info.get('model_type', 'Unknown')}")
    if 'language' in info:
        print(f"  语言: {info['language']}")
    if 'has_lora' in info:
        print(f"  使用 LoRA: ✅")
    print(f"{'='*60}\n")


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("🧪 测试模型库\n")
    
    # 测试创建 CLIP 模型
    try:
        print("测试 1: CLIP-ViT-B/32")
        model = create_model('clip-vit-b32', device='cpu')
        print_model_info(model)
    except Exception as e:
        print(f"❌ 测试失败: {e}\n")
    
    # 列出所有可用模型
    print("\n📚 所有可用模型:")
    for model_name in list_available_models():
        print(f"  - {model_name}")
    
    print("\n✅ 模型库测试完成")