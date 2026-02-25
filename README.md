---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- image-text-retrieval
---



# BLIP2-Qformer


## 简介 Brief Introduction

首个开源的中文BLIP2模型。我们遵循BLIP2的实验设置，采用itc、itm、lm损失，基于2亿中文图文对训练5个epoch，得到第一个中文版本的blip2模型。

The first open source Chinese BLIP2. We follow the experimental setup of BLIP2, we adopted itc, itm and lm losses, trained 5 epochs based on 200 million Chinese image pairs, and obtained the first Chinese version of BLIP2. 


### 下游效果 Performance


**Zero-Shot image-to-text-retrieval**

|  model   |  COCO-CN | Flickr30k-CN| 
|  ----  | ----  | ---- | 
| cn_clip  | 60.4 | 80.2| 
| cn_blip2(ours)  | 70.3 | 85.7| 

**Zero-Shot text-to-image-retrieval**

|  model   | COCO-CN | Flickr30k-CN | 
|  ----  | ----  | ---- | 
| cn_clip  | 64.0 | 68.0| 
| cn_blip2(ours)  | 71.4 | 70.46| 

## 使用 Usage

```bash
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download('xiajinpeng123/BLIP2-Chinese',revision='v1.0.0')
import os
os.chdir(model_path)
import sys
sys.path.insert(0, model_path)
import ms_wrapper
from modelscope.pipelines import pipeline
img = [f"{model_path}/test1.jpg",f"{model_path}/test3.jpg"]
txt=["两台汽车","白色标记","两辆汽车停在公路上","两只小鸟在树上"]
input_dict=dict()
input_dict['img']=img
input_dict['text']=txt
weight_path = f"{model_path}/checkpoint_04.pth"

inference = pipeline('image-text-retrieval', model='xiajinpeng123/BLIP2-Chinese',model_revision='v1.0.0', weight_path=weight_path,device="cuda") # GPU环境可以设置为True
output = inference(input_dict)

print(output)
```

```bash
 git clone https://www.modelscope.cn/xiajinpeng123/BLIP2-Chinese.git
```

## 使用方式及场景
### 使用方式：

- 对输入的图像、文本数据进行特征提取
### 使用场景:

- 通用的图文跨模态检索任务
- 通用图文特征提取器

## 模型局限性以及可能的偏差
- 训练数据集自身有局限，有可能产生一些偏差，请用户自行评测后决定如何使用。

#### 如果喜欢，敬请下载收藏！
