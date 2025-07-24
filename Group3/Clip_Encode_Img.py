# Clip_Encode_Img.py
import os
import torch
import clip
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

class CLIP_encoder:
    def __init__(self, model_name="ViT-L/14@336px", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def Vit_336px_encode(self, image_path):
        """
        对输入 JPG 图像进行 CLIP 编码，返回特征向量 [1, 768/1024]
        """
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # [1, 3, 336, 336]
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化
        return image_features  # shape: [1, 768] or [1, 1024]

    def show(self, encoding, topk=5, save_path=None):
        """
        打印特征维度、前topk维，并将整个特征保存为一张灰度图
        """
        encoding = encoding.detach().cpu().squeeze(0)  # [D]
        print(f"[INFO] 特征 shape: {encoding.shape}")  # e.g. torch.Size([1024])
        print(f"[INFO] 前 {topk} 维值: {encoding[:topk].numpy()}")

        # 可视化为灰度图
        plt.figure(figsize=(10, 1))
        plt.imshow(encoding.unsqueeze(0), cmap='gray', aspect='auto')
        plt.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"[INFO] 特征图已保存到: {save_path}")
        plt.close()

# ========== 示例调用 ==========
if __name__ == "__main__":
    encoder = CLIP_encoder()
    img_path = "/home/lumina/lumina/Jiawei/extracted_images/right_camera_frame000.jpg"
    feat = encoder.Vit_336px_encode(img_path)
    save_path = "/home/lumina/lumina/Jiawei/extracted_images/End/clip_feature.jpg"
    encoder.show(feat, topk=8, save_path=save_path)
