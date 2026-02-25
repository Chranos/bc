import os
import pdb

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
import numpy as np
import torch.nn.functional as F


# 使用任务+名称注册一个预处理器
@MODELS.register_module('image-text-retrieval', module_name='BLIP2_C')
class MaaSTemplateModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.device = kwargs.get('device')
        self.model = self.init_model(**kwargs)
        self.model.to(self.device)

    def forward(self, input_tensor, **forward_params):
        # raise NotImplementedError('model inference')
        txt_inputs=input_tensor['txt_inputs']
        img_inputs=input_tensor['img_inputs']

        text_embeds = []
        for txt_input in txt_inputs:
            txt_feat = self.model.forward_text(txt_input.to(self.device))
            text_embed = F.normalize(self.model.text_proj(txt_feat)) # 1*256
            text_embeds.append(text_embed)
        text_embeds = torch.cat(text_embeds, dim=0)

        image_embeds=[]
        for img_input in img_inputs:
            image_feat, vit_feat = self.model.forward_image(img_input.to(self.device))
            image_embed = F.normalize(self.model.vision_proj(image_feat),dim=-1)   # 1*32*256
            image_embeds.append(image_embed)

        sims_matrix = []
        for image_embed in image_embeds:
            sim_q2t = image_embed @ text_embeds.t()  # 1*32*1
            sim_i2t, _ = sim_q2t.max(0)
            out_result = torch.max(sim_i2t,0)
            sims_matrix.append(out_result.values)
        sims_matrix = torch.stack(sims_matrix, dim=0)
        return sims_matrix

    def init_model(self, **kwargs):
        """Provide default implementation based on TorchModel and user can reimplement it.
            include init model and load ckpt from the model_dir, maybe include preprocessor
            if nothing to do, then return lambdx x: x
        """
        # raise NotImplementedError('init_model')
        # return lambda x: x

        weight_path = kwargs.get('weight_path')

        # 创建模型
        if not os.path.isfile(weight_path):
            weight_path = os.path.join(self.model_dir, weight_path)
        model=BLIP2()
        checkpoint = torch.load(weight_path, map_location='cpu')

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

# 使用领域+名称注册一个预处理器
@PREPROCESSORS.register_module('multi-modal', module_name='my-custom-preprocessor')
class MaaSTemplatePreprocessor(Preprocessor):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.trainsforms = self.init_preprocessor(**kwargs)
        img_size=224
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, input):
        # 处理文本
        tokenizer=Blip2Base.init_tokenizer()
        text_inputs=[]
        for text in input['text']:
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            )
            text_inputs.append(text_input)
        # 处理图片
        image_inputs = []
        for img in input['img']:
            # image = Image.open(img).convert('RGB')
            if img.startswith("https") or img.startswith("http"):
                image = Image.open(requests.get(img, stream=True).raw).convert('RGB')
            else:
                image = Image.open(img).convert('RGB')
            image = self.transform(image)   # image=3*224*224
            image=image.unsqueeze(0)
            image_inputs.append(image)  # todo

        out_dict=dict()
        out_dict['txt_inputs']=text_inputs
        out_dict['img_inputs']=image_inputs   # 1*32*768
        return out_dict

    def init_preprocessor(self, **kwarg):
        """ Provide default implementation based on preprocess_cfg and user can reimplement it.
            if nothing to do, then return lambdx x: x
        """
        # pdb.set_trace()
        return lambda x: x


# 使用任务+名称注册一个pipeline
@PIPELINES.register_module('image-text-retrieval', module_name='BLIP2-Qformer')
class MaaSTemplatePipeline(Pipeline):

    def __init__(self, model, preprocessor=None, **kwargs):
        """
        use `model` and `preprocessor` to create a custom pipeline for prediction
        Args:
            model: model id on modelscope hub.
            preprocessor: the class of method be init_preprocessor
        """
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or Model'
        # import pdb

        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model, **kwargs)

        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.eval()
        if preprocessor is None:
            preprocessor = MaaSTemplatePreprocessor()  #

        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
           Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass
    def _check_output(self, outputs):
        pass

    def forward(self, inputs, **forward_params):
        """ Provide default implementation using self.model and user can reimplement it
        """
        return super().forward(inputs, **forward_params)

    def postprocess(self, inputs):
        """ If current pipeline support model reuse, common postprocess
           code should be write here.        
           Args:
            inputs:  input data
        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """

        return inputs


# Tips: usr_config_path is the temporary save configuration location， after upload modelscope hub, it is the model_id
usr_config_path = '.'
config = Config({
    'framework': 'pytorch',
    'task': 'image-text-retrieval',
    "model":{
    "type": "BLIP2_C",
    "scale": 2,
      "weight_path": "checkpoint_04.pth",
          "half":True
       },
      "pipeline": {"type": "BLIP2-Qformer"}
})
config.dump('.' + '/configuration.json')


if __name__ == "__main__":
    from modelscope.models import Model
    from modelscope.pipelines import pipeline
    # model = Model.from_pretrained(usr_config_path)
    img = ["test3.jpg","test1.jpg"]
    txt=["两台汽车","白色标记","两辆汽车停在公路上","两只小鸟在树上"]
    input_dict=dict()
    input_dict['img']=img
    input_dict['text']=txt
    inference = pipeline('image-text-retrieval', model=usr_config_path,
                         device="cuda:4")

    output = inference(input_dict)
    print(output)
