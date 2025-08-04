import os
import urllib.request
import torch
import clip
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch.nn.functional as F

class CLIP_encoder:
    def __init__(self, model_name="ViT-L/14@336px", device=None):
        self.device = device or torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
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

    def save_image_from_tensor(self, tensor, save_path1):
        if tensor.ndim == 3 and tensor.shape[0] == 1024:
            tensor = np.transpose(tensor, (1, 2, 0))

        if tensor.ndim == 3 and tensor.shape[2] == 3:
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
            tensor = (tensor * 255).astype(np.uint8)

            os.makedirs(os.path.dirname(save_path1), exist_ok=True)
            Image.fromarray(tensor).save(save_path1)
            print(f"[INFO] 特征图已保存为 JPG 到: {save_path1}")
        else:
            print(f"[ERROR] 无法保存图像，张量的形状不符合要求: {tensor.shape}")

    def show(self, image_path, save_path1):
        feat = self.Vit_336px_CLIPFeatureExtractor(image_path).detach().cpu()
        print(f"[DEBUG] 原始特征 shape: {feat.shape}")  # (1, 1024, 24, 24)
        # 双线性插值到 [1, 1024, 240, 320]
        feat_resized = F.interpolate(feat, size=(240, 320), mode='bilinear', align_corners=False)
        
        feat_resized = feat_resized.squeeze(0).numpy() # [1024, 240, 320]
        print(f"[DEBUG] 插值后的特征 shape: {feat_resized.shape}")
        C, H, W = feat_resized.shape  # [1024, 240, 320]
        # 展平空间维度，准备做 PCA：[H*W, C]
        feat_flat = feat_resized.reshape(C, -1).T  # [76800, 1024]
        
        pca = PCA(n_components=3)
        feat_resized_rgb = pca.fit_transform(feat_flat) # [76800, 3]
        feat_resized_rgb = feat_resized_rgb.T.reshape(3, H, W) # 转换为 [3, 240, 320]
        
        # 转置为 [H, W, C] → [240, 320, 3]
        feat_resized_rgb = np.transpose(feat_resized_rgb, (1, 2, 0))
        self.save_image_from_tensor(feat_resized_rgb, save_path1)


# ========== 示例调用 ==========

# if __name__ == "__main__":
#     encoder = CLIP_encoder()
#     img_path = "/home/lumina/lumina/Jiawei/extracted_images/Extracted_img_front_camera_random/right_camera_frame000.jpg"
#     save_path1 = "/home/lumina/lumina/Jiawei/extracted_images/End/rgb_feature_map_240320.jpg"
#     save_path2 = "/home/lumina/lumina/Jiawei/extracted_images/End/rgb_feature_map.jpg"
#     encoder.show(img_path, save_path1)
    
