#!/bin/bash

echo "📦 安装模型库依赖..."

# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CLIP
pip install git+https://github.com/openai/CLIP.git

# Transformers (用于 Chinese-CLIP, BLIP)
pip install transformers

# PEFT (用于 LoRA)
pip install peft

# timm (用于 ViT)
pip install timm

# 其他工具
pip install pillow scikit-learn tqdm

echo "✅ 依赖安装完成！"