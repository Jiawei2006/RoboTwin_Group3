# ================================================
# Diffusion Policy for Point Cloud (DP3) - 主模块实现
# 实现核心：以条件一维U-Net为基础的动作生成扩散策略
# ================================================

# ===== 🧩 标准库与第三方库导入 =====
from typing import Dict
import math, time, copy, pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint

# ===== 📦 项目依赖模块导入 =====
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder

# ===== 🚀 DP3 策略模型类 =====
class DP3(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon, n_action_steps, n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_type = condition_type

        # ===== 💡 动作维度解析 =====
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        # ===== 🧠 观测编码器定义 =====
        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])
        obs_encoder = (DP3Encoder(
            # Crop + FPS (Farthest Point Sampling) 的预处理；
            # Compact 3D Representations by MLP + MaxPool
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        ))

        # ===== 🌀 条件 U-Net 定义（Diffusion 模型） =====
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        # ===== 🧱 模型组件赋值 =====
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # ===== ✅ 打印参数 =====
        cprint(f"[DP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DP3] pointnet_type: {self.pointnet_type}", "yellow")
        print_params(self)

    # ========== 🔁 采样流程（评估阶段） ========== #
    def conditional_sample(self, condition_data, condition_mask,
                           condition_data_pc=None, condition_mask_pc=None,
                           local_cond=None, global_cond=None,
                           generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        # 初始化随机噪声轨迹
        trajectory = torch.randn_like(condition_data)
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        # 再次强制条件一致性
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ========== 🔮 推理阶段动作预测 ========== #
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3] # Single-view Point Cloud 输入
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T, Da, Do = self.horizon, self.action_dim, self.obs_feature_dim

        device, dtype = self.device, self.dtype
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = (nobs_features.reshape(B, To, -1)
                           if "cross_attention" in self.condition_type
                           else nobs_features.reshape(B, -1))
            cond_data = torch.zeros((B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(B, To, -1)
            cond_data = torch.zeros((B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        return {"action": action_pred[:, start:end], "action_pred": action_pred}

    # ========== 🧪 训练阶段损失计算 ========== #
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        B, T = nactions.shape[:2]
        trajectory = cond_data = nactions
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = (nobs_features.reshape(B, self.n_obs_steps, -1)
                           if "cross_attention" in self.condition_type
                           else nobs_features.reshape(B, -1))
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(B, T, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        # ===== 🎯 损失目标构造 =====
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t = self.noise_scheduler.alpha_t[timesteps].unsqueeze(-1).unsqueeze(-1)
            sigma_t = self.noise_scheduler.sigma_t[timesteps].unsqueeze(-1).unsqueeze(-1)
            target = alpha_t * noise - sigma_t * trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()

        return loss, {"bc_loss": loss.item()}
