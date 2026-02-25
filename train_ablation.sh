#!/bin/bash

echo "🔬 开始批量消融实验"

experiments=(
    "itc_only"
    "scene_only"
    # "itc_scene_equal"
    # "itc_scene_2_8"
    # "itc_scene_8_2"
    # "itc_itm"
    # "itc_scene_itm"
    # "itc_scene_itm_equal"
)

for exp in "${experiments[@]}"; do
    echo ""
    echo "========================================="
    echo "开始实验: $exp"
    echo "========================================="
    
    python train_ablation.py \
        --ablation "$exp" \
        --epochs 20 \
        --batch_size 32 \
        --lr 1e-4 \
        --device cuda:4
    
    echo "✅ 完成: $exp"
    echo ""
done

echo "🎉 所有消融实验完成！"