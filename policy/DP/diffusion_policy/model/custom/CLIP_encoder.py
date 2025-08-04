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
    def __init__(self, model_name="ViT-B/16", device=None):
        # import pdb; pdb.set_trace()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        self.model_dir = "/home/lumina/lumina/Jiawei/RoboTwin/Group3/models"
        os.environ["TORCH_HOME"] = self.model_dir
        self._ensure_model_download(model_name)

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        

    def _ensure_model_download(self, model_name):
        model_url_map = {
            "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
            "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
        }
        model_file_map = {
            "ViT-L/14@336px": "ViT-L-14-336px.pt",
            "ViT-B/16": "ViT-B-16.pt"
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
        
    def Vit_336px_CLIPFeatureExtractor_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        输入：np.ndarray 图像，形状为 [H, W, 3]，dtype=uint8
        输出：降维后的 RGB 特征图，形状为 [3, 240, 320]
        """
        image = Image.fromarray(image_array).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device).to(torch.float16)

        with torch.no_grad():
            visual = self.model.visual
            feat = visual.conv1(image_input)  # [1, 1024, 24, 24]
            feat_resized = F.interpolate(feat, size=(240, 320), mode='bilinear', align_corners=False)
            feat_resized = feat_resized.squeeze(0).cpu().numpy()  # [1024, 240, 320]

            feat_flat = feat_resized.reshape(feat_resized.shape[0], -1).T  # [76800, 1024]
            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feat_flat).T.reshape(3, 240, 320)  # [3, 240, 320]
            return feat_pca.astype(np.float32)


    def save_featmap_from_tensor(self, tensor, save_path1):
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

    def extract_feat_map(self, image_path):
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
        
        return feat_resized_rgb

    def save_feat_map(self, image_path, save_path1):
        feat_resized_rgb = self.extract_feat_map(image_path)
        self.save_featmap_from_tensor(feat_resized_rgb, save_path1)
   
    



    
