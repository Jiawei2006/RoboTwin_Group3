import torch
import torchvision


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    # resnet_new = torch.nn.Sequential(
    #     resnet,
    #     torch.nn.Linear(512, 128)
    # )
    # return resnet_new
    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model



import torch
import torchvision
import os
import sys

# 添加CLIP_encoder的路径
sys.path.append("/home/lumina/lumina/Jiawei/RoboTwin/Group3")
from extract_ob_B import CLIP_encoder

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_CLIP_Vit(name="ViT-L/14@336px", device=None, **kwargs):
    """
    获取CLIP视觉编码器，接口与torchvision.models保持一致
    
    name: CLIP模型名称
    device: 设备
    """
    # 创建CLIP编码器实例
    clip_encoder = CLIP_encoder(model_name=name, device=device, **kwargs)
    
    # 创建兼容torchvision接口的包装器
    class CLIPVisionModel(torch.nn.Module):
        def __init__(self, clip_encoder):
            super().__init__()
            self.clip_encoder = clip_encoder
            # 添加一个"fc"层以保持与ResNet接口一致（虽然不会被使用）
            self.fc = torch.nn.Identity()
            
        def forward(self, x):
            """
            注意：CLIP编码器需要图片路径而不是张量
            在实际使用中，您需要修改数据加载部分以传递图片路径
            """
            # 这里只是一个占位实现，实际使用时需要根据数据格式调整
            raise NotImplementedError(
                "CLIP编码器需要图片路径作为输入，而不是张量。"
                "请修改数据加载器以传递图片路径，或使用clip_encoder.forward()方法直接处理路径。"
            )
            
        def encode_image_paths(self, image_paths):
            """
            专门用于处理图片路径的接口
            
            image_paths: str (单路径) 或 list of str (多路径)
            返回: [B, 512] 特征向量
            """
            if isinstance(image_paths, str):
                # 单路径或目录路径
                return self.clip_encoder.forward(image_paths)
            elif isinstance(image_paths, (list, tuple)):
                # 多路径处理
                features = []
                for path in image_paths:
                    feat = self.clip_encoder.forward_single(path)
                    features.append(feat)
                return torch.cat(features, dim=0)  # [B, 512]
            else:
                raise ValueError(f"不支持的输入格式: {type(image_paths)}")
    
    return CLIPVisionModel(clip_encoder)

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model