import os
import torch
import torch.nn as nn
from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS

from models.blip2_qformer import Blip2Qformer as BLIP2
from models.blip2 import Blip2Base
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import requests


# 注册图像描述生成模型
@MODELS.register_module('image-captioning', module_name='BLIP2_Caption')
class ImageCaptioningModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model(**kwargs)
        self.model.to(self.device)
        
        # 初始化 tokenizer 用于解码生成的文本
        self.tokenizer = Blip2Base.init_tokenizer()
        print(f"✅ 模型已加载到: {self.device}")

    def forward(self, input_tensor, **forward_params):
        """生成图像描述"""
        img_inputs = input_tensor['img_inputs']
        
        # 提取生成参数（只使用模型支持的参数）
        max_length = forward_params.get('max_length', 30)
        min_length = forward_params.get('min_length', 10)
        num_beams = forward_params.get('num_beams', 3)
        top_p = forward_params.get('top_p', 0.9)
        repetition_penalty = forward_params.get('repetition_penalty', 1.0)
        use_nucleus_sampling = forward_params.get('use_nucleus_sampling', False)
        
        captions = []
        with torch.no_grad():
            for img_input in img_inputs:
                img_input = img_input.to(self.device)
                
                # 调用模型的 generate 方法生成描述
                # BLIP2 模型需要输入字典格式
                samples = {"image": img_input}
                
                # 使用模型的生成方法（只传递支持的参数）
                output_ids = self.model.generate(
                    samples,
                    use_nucleus_sampling=use_nucleus_sampling,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
                # 解码生成的 token IDs
                if isinstance(output_ids, torch.Tensor):
                    # 如果返回的是张量
                    caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                elif isinstance(output_ids, list):
                    # 如果返回的是列表（BLIP2通常返回列表）
                    caption = output_ids[0] if output_ids else "无法生成描述"
                else:
                    caption = str(output_ids)
                
                captions.append(caption.strip())
        
        return captions

    def init_model(self, **kwargs):
        """初始化模型并加载权重"""
        weight_path = kwargs.get('weight_path')

        if not os.path.isfile(weight_path):
            weight_path = os.path.join(self.model_dir, weight_path)
        
        # 创建 BLIP2 模型
        model = BLIP2()
        checkpoint = torch.load(weight_path, map_location='cpu')

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model


# 注册图像描述预处理器
@PREPROCESSORS.register_module('multi-modal', module_name='caption-preprocessor')
class CaptionPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 图像预处理（与 BLIP2 训练时一致）
        img_size = 224
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, input):
        """预处理输入图片"""
        # 支持单张图片或图片列表
        if isinstance(input, str):
            images = [input]
        elif isinstance(input, list):
            images = input
        elif isinstance(input, dict) and 'img' in input:
            images = input['img'] if isinstance(input['img'], list) else [input['img']]
        else:
            raise ValueError(f"不支持的输入格式: {type(input)}")
        
        # 处理每张图片
        image_inputs = []
        for img in images:
            if isinstance(img, str):
                # 支持 URL 或本地路径
                if img.startswith("https") or img.startswith("http"):
                    image = Image.open(requests.get(img, stream=True).raw).convert('RGB')
                else:
                    image = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                image = img.convert('RGB')
            else:
                raise ValueError(f"不支持的图片格式: {type(img)}")
            
            # 应用变换
            image = self.transform(image)  # [3, 224, 224]
            image = image.unsqueeze(0)     # [1, 3, 224, 224]
            image_inputs.append(image)
        
        return {'img_inputs': image_inputs}


# 注册图像描述生成管道
@PIPELINES.register_module('image-captioning', module_name='BLIP2-Caption')
class ImageCaptioningPipeline(Pipeline):

    def __init__(self, model, preprocessor=None, **kwargs):
        """初始化图像描述生成管道"""
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or Model'

        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model, **kwargs)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        
        pipe_model.eval()
        
        if preprocessor is None:
            preprocessor = CaptionPreprocessor()

        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """分离预处理、前向和后处理参数"""
        # 生成参数传递给 forward（只传递支持的参数）
        forward_params = {}
        supported_params = ['max_length', 'min_length', 'num_beams', 
                           'top_p', 'repetition_penalty', 'use_nucleus_sampling']
        
        for key in supported_params:
            if key in pipeline_parameters:
                forward_params[key] = pipeline_parameters[key]
        
        return {}, forward_params, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, inputs, **forward_params):
        """执行前向推理"""
        return super().forward(inputs, **forward_params)

    def postprocess(self, inputs):
        """后处理：格式化输出"""
        return inputs


# 配置文件
usr_config_path = '.'
config = Config({
    'framework': 'pytorch',
    'task': 'image-captioning',
    "model": {
        "type": "BLIP2_Caption",
        "weight_path": "checkpoint_04.pth",
        "half": False
    },
    "pipeline": {"type": "BLIP2-Caption"}
})
config.dump('.' + '/configuration.json')


# 使用示例
if __name__ == "__main__":
    from modelscope.pipelines import pipeline
    
    print("\n" + "="*60)
    print("🖼️  BLIP2 中文图像描述生成")
    print("="*60 + "\n")
    
    # 创建推理管道
    caption_pipeline = pipeline(
        'image-captioning', 
        model=usr_config_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    

    image_paths = ["test1.jpg", "test3.jpg"]
    existing_images = [img for img in image_paths if os.path.exists(img)]
    input_dict = {'img': existing_images if existing_images else ["test1.jpg"]}


    
    if existing_images:
        # 使用 beam search（更准确）
        captions_beam = caption_pipeline(
            existing_images[1],
            num_beams=5,
            max_length=30,
            repetition_penalty=1.2
        )
        print(f"📷 图片: {existing_images[1]}")
        print(f"📝 Beam Search: {captions_beam[0]}")
        

    
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60 + "\n")