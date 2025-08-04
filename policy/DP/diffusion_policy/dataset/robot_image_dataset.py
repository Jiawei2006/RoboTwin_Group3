import pdb
import os
from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.custom.CLIP_encoder import CLIP_encoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.custom.CNN_6_to_3 import CNN_6_to_3


class RobotImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):
        super().__init__()
        # pdb.set_trace()
        # 当设置CUDA_VISIBLE_DEVICES时，PyTorch会重新映射设备索引
        # 例如CUDA_VISIBLE_DEVICES=6时，实际应该使用cuda:0
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print(f"[INFO] RobotImageDataset using device: {device}")
        self.clip_encoder = CLIP_encoder(device=device)
        self.feature_save_root = "/home/lumina/lumina/Jiawei/RoboTwin/policy/DP/data/move_can_pot-demo_clean-50.zarr/data/head_camera/CLIP_extracted"
        os.makedirs(self.feature_save_root, exist_ok=True)
        self.save_counter = 0

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["head_camera", "state", "action"],
        )
        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio, seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        data = {
            "obs": {
                "head_cam": head_cam,
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam_rgb = samples["head_camera"].to(device, non_blocking=True).float() / 255.0
        B, T, C, H, W = head_cam_rgb.shape
        head_cam_clip = []

        # for b in range(B):
        #     clip_batch = []
        #     for t in range(T):
        #         img_np = (head_cam_rgb[b, t].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            
        #         # 提取特征
        #         feat = self.clip_encoder.Vit_336px_CLIPFeatureExtractor_from_array(img_np)
        #         feat = np.transpose(feat, (1, 2, 0))  # (3, 240, 320) → (240, 320, 3)
        #         # import pdb; pdb.set_trace()
        #         print("[INFO] feat.shape: ", feat.shape)
                
        #         # 保存特征图
        #         save_raw_name = f"b{b}_t{t}_raw{self.save_counter}.jpg"
        #         save_raw_path = os.path.join(self.feature_save_root, save_raw_name)
        #         self.clip_encoder.save_featmap_from_tensor(img_np, save_raw_path)
        #         save_name = f"b{b}_t{t}_extracted{self.save_counter}.jpg"
        #         save_path = os.path.join(self.feature_save_root, save_name)
        #         self.clip_encoder.save_featmap_from_tensor(feat, save_path)
        #         self.save_counter += 1
                
        #         feat = np.transpose(feat, (2, 0, 1))  # (H, W, C) → (C, H, W)
        #         clip_batch.append(torch.from_numpy(feat))
                
        #     head_cam_clip.append(torch.stack(clip_batch))

        for b in range(B):
            clip_batch = []
            for t in range(T):
                img_np = (head_cam_rgb[b, t].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                # 构造缓存路径（.npy 格式特征图）
                save_name = f"b{b}_t{t}_extracted{self.save_counter}.npy"
                save_path = os.path.join(self.feature_save_root, save_name)

                # import pdb; pdb.set_trace()
                # print(f"[DEBUG] 检查路径: {save_path} | 是否存在: {os.path.exists(save_path)}")

                if os.path.exists(save_path):
                    # 已有缓存，直接加载
                    feat = np.load(save_path)
                    print(f"[INFO] Loaded cached feature from {save_path}")
                else:
                    # 提取特征
                    feat = self.clip_encoder.Vit_336px_CLIPFeatureExtractor_from_array(img_np)
                    feat = np.transpose(feat, (1, 2, 0))  # (3, 240, 320) → (240, 320, 3)
                    print("[INFO] feat.shape: ", feat.shape)

                    # 保存可加载的中间特征数据
                    np.save(save_path, feat)
                    print(f"[INFO] Saved extracted feature to {save_path}")
                
                self.save_counter += 1
                print(f"[INFO] save_counter {self.save_counter}")

                # 转换为 torch 格式
                feat = np.transpose(feat, (2, 0, 1))  # (H, W, C) → (C, H, W)
                clip_batch.append(torch.from_numpy(feat))

            head_cam_clip.append(torch.stack(clip_batch))


        head_cam_clip = torch.stack(head_cam_clip)  # [B, T, 3, H, W]
        print(f"[DEBUG] head_cam_rgb.shape: {head_cam_rgb.shape}")
        print(f"[DEBUG] head_cam_clip.shape: {head_cam_clip.shape}")
        head_cam_concat = torch.cat([head_cam_rgb, head_cam_clip.to(device)], dim=2)  # [B, T, 6, H, W]

        # 初始化一次 CNN 模块（建议写在 __init__ 中复用）
        cnn_6to3 = CNN_6_to_3().to(device)

        # 扁平化张量 [B, T, 6, H, W] → [B*T, 6, H, W]
        B, T, _, H, W = head_cam_concat.shape
        head_cam_6 = head_cam_concat.view(B*T, 6, H, W)

        # 转换为 3 通道
        head_cam_3 = cnn_6to3(head_cam_6)

        # 再 reshape 回 [B, T, 3, H, W]
        head_cam = head_cam_3.view(B, T, 3, H, W)
        
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": {
                # "head_cam": head_cam_concat,
                "head_cam": head_cam,
                "agent_pos": agent_pos,
            },
            "action": action,
        }


# batch sample utils
def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
