#!/bin/bash

echo "🚀 批量微调所有模型"

models=("clip-vit-b32" "clip-vit-l14" "resnet50" "vit-base")

for model in "${models[@]}"; do
    echo ""
    echo "="
    echo "开始微调: $model"
    echo "========================================="
    
    # 修改配置并运行
    python train_all.py \
        --model_name "$model" \
        --batch_size 64 \
        --epochs 10 \
        --device cuda:4
    
    echo "✅ $model 微调完成"
    echo ""
done

echo "🎉 所有模型微调完成！"