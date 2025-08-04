import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import os
import urllib.request
import clip
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import glob

class CLIP_encoder:
    def __init__(self, model_name="ViT-L/14@336px", device=None):
        self.device = device or torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        self.model_dir = "/home/lumina/lumina/Jiawei/RoboTwin/Group3/models"
        os.environ["TORCH_HOME"] = self.model_dir
        self._ensure_model_download(model_name)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        

    def _ensure_model_download(self, model_name):
        model_url_map = {
            "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
        }
        model_file_map = {
            "ViT-L/14@336px": "ViT-L-14-336px.pt"
        }

        if model_name not in model_file_map:
            raise ValueError(f"暂不支持模型: {model_name}")

        model_path = os.path.join(self.model_dir, "clip", model_file_map[model_name])
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"[INFO] 模型未缓存，正在下载到: {model_path}")
            urllib.request.urlretrieve(model_url_map[model_name], model_path)
            print("[INFO] 模型下载完成")
        else:
            print(f"[INFO] 已检测到本地模型缓存：{model_path}")

    def Vit_336px_CLIPFeatureExtractor(self, image_path):
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device).to(torch.float16)
        print(f"[INFO] preprocess.shape: ", image_input.shape) # [1, 3, 336, 336]
        with torch.no_grad():
            _ = self.model.encode_image(image_input)
            visual = self.model.visual
            x = visual.conv1(image_input)  # shape: [1, 1024, 24, 24]
            print(f"[INFO] 提取的特征图形状: {x.shape}")
            return x
        
    def pca_tensor(self, tensor):
        """
        将CLIP特征通过PCA降维转换为类似ResNet的输出格式
        
        输入: [1, 1024, 24, 24] 
        输出: [1, 512] (通过3个通道各512维特征组合)
        """
        # shape: [1, 1024, 24, 24] > [1, 1024, 24*24]
        # pca: [1, 1024, 24, 24] > [1, 3, 24*24]
        # pca: [1, 3, 24*24] > [1, 3, 512]
        # 拆分: [1, 3, 512] > [1, 512]*3
        
        # 确保输入是正确的形状
        assert tensor.shape == (1, 1024, 24, 24), f"期望输入形状 [1, 1024, 24, 24]，实际得到 {tensor.shape}"
        
        # 1. [1, 1024, 24, 24] > [1, 1024, 24*24]
        batch_size, channels, height, width = tensor.shape
        feat = tensor.view(batch_size, channels, -1)  # [1, 1024, 576]
        
        # 2. 转换维度以进行PCA: [1, 1024, 576] > [576, 1024]
        feat = feat.squeeze(0).permute(1, 0)  # [576, 1024]
        
        # 3. 应用PCA降维: [576, 1024] > [576, 3]
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat.cpu().numpy())  # [576, 3]
        
        # 4. 转换为PyTorch张量
        feat_pca = torch.from_numpy(feat_pca).float().to(tensor.device)
        
        # 5. 重塑并降维到512: [576, 3] > 每个通道降维到512维
        # 分离三个通道: [576, 3] -> 3 * [576]
        channel1 = feat_pca[:, 0]  # [576]
        channel2 = feat_pca[:, 1]  # [576]
        channel3 = feat_pca[:, 2]  # [576]
        print("[INFO] 单通道tensor:", channel1.shape)
        
        # 每个通道通过线性层降维到512
        if not hasattr(self, 'channel_proj'):
            self.channel_proj = torch.nn.Linear(576, 512).to(tensor.device)
        
        # 应用投影
        feat1 = self.channel_proj(channel1.unsqueeze(0))  # [1, 512]
        feat2 = self.channel_proj(channel2.unsqueeze(0))  # [1, 512]
        feat3 = self.channel_proj(channel3.unsqueeze(0))  # [1, 512]
        
        # 合并三个通道的特征
        combined_feat = torch.cat([feat1, feat2, feat3], dim=0)  # [3, 512]
        return combined_feat

    def forward_single(self, image_path):
        """
        处理单张图片
        """
        feat = self.Vit_336px_CLIPFeatureExtractor(image_path)
        print("[INFO] feat.shape: ", feat.shape) # [1, 1024, 24, 24]
        resized_feat = self.pca_tensor(feat)
        print("[INFO] resized_feat.shape: ", resized_feat.shape) # [3, 512]
        
        return resized_feat
    
    def forward_batch(self, images_dir):
        """
        批量处理目录中的所有图片
        
        输入: images_dir - 包含图片的目录路径
        输出: [B, 512] - B是图片数量
        """
        # 获取目录中所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for extension in image_extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, extension)))
            image_paths.extend(glob.glob(os.path.join(images_dir, extension.upper())))
        
        if not image_paths:
            raise ValueError(f"在目录 {images_dir} 中未找到图片文件")
        
        print(f"[INFO] 找到 {len(image_paths)} 张图片")
        
        # 按文件名排序以确保顺序一致
        image_paths.sort()
        
        batch_features = []
        for i, image_path in enumerate(image_paths):
            print(f"[INFO] 处理第 {i+1}/{len(image_paths)} 张图片: {os.path.basename(image_path)}")
            try:
                feat = self.forward_single(image_path)
                batch_features.append(feat)
            except Exception as e:
                print(f"[ERROR] 处理图片 {image_path} 时出错: {e}")
                continue
        
        if not batch_features:
            raise RuntimeError("未能处理任何图片")
        
        # 将所有特征堆叠成批次
        batch_feat = torch.cat(batch_features, dim=0)  # [B, 512]
        print(f"[INFO] 批量特征形状: {batch_feat.shape}") # [462, 512]
        
        return batch_feat

    def forward(self, image_input):
        """
        统一接口，支持单张图片路径或目录路径
        
        输入: image_input - 单张图片路径(str) 或 图片目录路径(str)
        输出: [B, 512] - B是图片数量
        """
        if os.path.isfile(image_input):
            # 单张图片
            feat = self.forward_single(image_input)
            return feat
        elif os.path.isdir(image_input):
            # 图片目录
            return self.forward_batch(image_input)
        else:
            raise ValueError(f"无效的输入路径: {image_input}")

# ========== 示例调用 ==========

if __name__ == "__main__":
    encoder = CLIP_encoder()
    
    # # 处理单张图片
    # img_path = "/home/lumina/lumina/Jiawei/extracted_images/Extracted_img_front_camera_random/right_camera_frame000.jpg"
    # print("=== 处理单张图片 ===")
    # single_feat = encoder.forward(img_path)
    # print(f"单张图片特征形状: {single_feat.shape}")
    
    # 批量处理目录中的所有图片
    imgs_path = "/home/lumina/lumina/Jiawei/extracted_images/Extracted_img_right_camera"
    print("\n=== 批量处理图片目录 ===")
    batch_feat = encoder.forward(imgs_path)
    print(f"批量图片特征形状: {batch_feat.shape}")