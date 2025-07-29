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
        print(f"[INFO] preprocess.shape: ", image_input.shape)
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
        feat = self.Vit_336px_CLIPFeatureExtractor(image_path).detach().cpu().numpy()
        print(f"[DEBUG] 原始特征 shape: {feat.shape}")  # (1, 1024, 24, 24)

        feat = feat.reshape(1024, -1).T  # → shape: (576, 1024)
        print(f"[DEBUG] PCA前特征 shape: {feat.shape}")

        pca = PCA(n_components=3)
        feat_rgb = pca.fit_transform(feat)  # → shape: (576, 3)
        print(f"[DEBUG] PCA后特征 shape: {feat_rgb.shape}")

        rgb = (feat_rgb - feat_rgb.min()) / (np.ptp(feat_rgb, axis=0) + 1e-8)
        rgb_img = rgb.reshape(24, 24, 3)  # → shape: (24, 24, 3)
        print(f"[Info] rgb_img RGB图 shape: {rgb_img.shape}")
        self.save_image_from_tensor(rgb_img,save_path2)

        # Step 4: 插值到 (64, 64, 3)
        rgb_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, 24, 24]
        rgb_resized = F.interpolate(rgb_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        rgb_resized = rgb_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [64, 64, 3]
        self.save_image_from_tensor(rgb_resized, save_path1)
        print(f"[Info] rgb_resized RGB图 shape: {rgb_resized.shape}")


# ========== 示例调用 ==========

if __name__ == "__main__":
    encoder = CLIP_encoder()
    img_path = "/home/lumina/lumina/Jiawei/extracted_images/Extracted_img_front_camera_random/right_camera_frame000.jpg"
    save_path1 = "/home/lumina/lumina/Jiawei/extracted_images/End/rgb_feature_map_resized.jpg"
    save_path2 = "/home/lumina/lumina/Jiawei/extracted_images/End/rgb_feature_map.jpg"
    encoder.show(img_path, save_path1)
    
