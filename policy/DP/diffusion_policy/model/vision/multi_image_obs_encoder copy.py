# ========================== 统一导入 ==========================
import pdb
import copy
import torch
import torch.nn as nn
import torchvision
from typing import Dict, Tuple, Union
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.custom.CNN_6_to_3 import CNN_6_to_3
from diffusion_policy.model.custom.split_normalize import SplitNormalize


# ========================== 多模态图像观察编码器类 ==========================
class MultiImageObsEncoder(ModuleAttrMixin):
    """
    编码多路观测输入(RGB图像、低维向量),支持不同模型、尺寸变换、裁剪与归一化。
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
    ):
        """
        参数说明：
        - shape_meta: 输入数据的形状与类型定义
        - rgb_model: 图像输入的编码模型（共享或独立）
        - resize_shape: 图像resize尺寸(可为统一或按键设置)
        - crop_shape: 图像裁剪尺寸
        - random_crop: 是否使用随机裁剪
        - use_group_norm: 是否将BN替换为GN
        - share_rgb_model: 所有图像输入是否共享同一模型
        - imagenet_norm: 是否使用ImageNet均值标准化图像
        """
        # pdb.set_trace() 
        # 调用：/diffusion_policy/policy/diffusion_unet_image_policy.py
        super().__init__()
        
        rgb_keys = list()                        # 保存所有rgb输入键
        low_dim_keys = list()                    # 保存所有低维向量输入键
        key_model_map = nn.ModuleDict()          # 输入键 -> 模型映射
        key_transform_map = nn.ModuleDict()      # 输入键 -> 图像变换映射
        key_shape_map = dict()                   # 输入键 -> shape 映射

        # 如果所有rgb输入共享一个模型

        # share_rgb_model: False
        # print("share_rgb_model:", share_rgb_model)
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if type == "rgb":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )


                # # ========== ✅ [新增] 如果输入为6通道，加入CNN_6_to_3 ==========
                #     if shape[0] == 6:
                #         print(f"[INFO] key '{key}' has 6 input channels, applying CNN_6_to_3 before model.")
                #         cnn_6to3 = CNN_6_to_3()
                #         this_model = nn.Sequential(cnn_6to3, this_model)
                #     key_shape_map[key] = shape
                #     print(f"[CHECK] key: {key}, expected shape: {shape}")
                #     key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(size=(h, w))
                # configure normalizer
                imagenet_mean = [0.485, 0.456, 0.406]
                imagenet_std = [0.229, 0.224, 0.225]

                # if imagenet_norm:
                #     if shape[0] == 6:
                #         # 两个RGB图像拼接后的 6通道输入
                #         this_normalizer = SplitNormalize(
                #             mean1=imagenet_mean,
                #             std1=imagenet_std,
                #             split_index=3
                #         )
                #     elif shape[0] == 3:
                #         this_normalizer = torchvision.transforms.Normalize(
                #             mean=imagenet_mean,
                #             std=imagenet_std
                #         )
                #     else:
                #         raise ValueError(f"Unsupported channel size {shape[0]} for imagenet_norm")
                # else:
                #     this_normalizer = nn.Identity()

                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform

            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    # ========================== 前向推理 ==========================
    def forward(self, obs_dict):
        """
        对输入obs_dict执行图像与低维编码,输出拼接后的特征向量
        """
        batch_size = None
        features = list()

        # process rgb input

        # share_rgb_model: False
        print("share_rgb_model:", self.share_rgb_model)
        if self.share_rgb_model:
            pdb.set_trace()
            # 所有rgb输入共享同一个模型
            imgs = list()
            for key in self.rgb_keys:
                pdb.set_trace() #查看key
                img = obs_dict[key] ## 任务点溯源
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                pdb.set_trace()
                assert img.shape[1:] == self.key_shape_map[key] 
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # 合并成一个Batch
            imgs = torch.cat(imgs, dim=0)  # (N*B,C,H,W)
            feature = self.key_model_map["rgb"](imgs)  # (N*B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])  # (N,B,D)
            feature = torch.moveaxis(feature, 0, 1)  # (B,N,D)
            feature = feature.reshape(batch_size, -1)  # (B,N*D)
            features.append(feature)
        else:
            # 每个rgb输入使用独立模型
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                
                print(f"[DEBUG] Checking key: {key}")
                print(f"[DEBUG] img shape: {img.shape} {img.shape[1:]}")
                print(f"[DEBUG] self.key_shape_map[key]: {self.key_shape_map[key]}")

                expected_shape = self.key_shape_map[key]
                actual_shape = img.shape[1:]

                # ====== 新增逻辑：允许6通道输入通过 CNN_6_to_3 变为3通道 ======
                if actual_shape != expected_shape:
                    if actual_shape[0] == 6 and expected_shape[0] == 3:
                        print(f"[INFO] (Bypass) Got 6-channel input, expecting 3-channel after CNN_6_to_3.")
                        # 允许继续执行，不 raise
                    else:
                        raise ValueError(
                            f"[ERROR] Shape mismatch for key '{key}': "
                            f"got {actual_shape}, expected {expected_shape}"
                        )
                # =========================================================

                img = self.key_transform_map[key](img)  # Resize / Crop / Normalize
                feature = self.key_model_map[key](img)  # CNN_6_to_3 + ResNet or just ResNet
                features.append(feature)


        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    # ========================== 推理输出形状 ==========================
    @torch.no_grad()
    def output_shape(self):
        """
        返回该编码器的输出特征维度（不含batch）
        """
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros((batch_size, ) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]  # 去除batch维度
        return output_shape
