from CLIP import clip
from torch import nn
import RoboTwin
import matplotlib.pyplot as plt
import matplotlib
import h5py
import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import io
import torch
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torchvision import transforms

plt.rcParams['axes.unicode_minus'] = False   
plt.rcParams['text.usetex'] = False   

# 6. 可视化
def PCA_visualize(feature, H, W, return_res=False):
    """
    对特征图进行PCA可视化
    """
    feature_img_resized = F.interpolate(feature, 
                            size=(H, W), 
                            mode='bilinear', 
                            align_corners=True)
    feature_img_resized = feature_img_resized[0].permute(1, 2, 0)
    feature = feature_img_resized
    if feature.device != torch.device('cpu'):
        feature = feature.cpu()
    pca = PCA(n_components=3)
    tmp_feature = feature.reshape(-1, feature.shape[-1]).detach().numpy()
    pca.fit(tmp_feature)
    pca_feature = pca.transform(tmp_feature)
    for i in range(3):  # min_max scaling
        pca_feature[:, i] = (pca_feature[:, i] - pca_feature[:, i].min()) / (pca_feature[:, i].max() - pca_feature[:, i].min())
    pca_feature = pca_feature.reshape(feature.shape[0], feature.shape[1], 3)
    print(f"PCA特征图形状: {pca_feature.shape}")
    show_img = Image.fromarray((pca_feature * 255).astype(np.uint8))
    if return_res:
        return pca_feature
    plt.imshow(pca_feature)
    plt.axis('off')
    plt.show()

def load_image(url, transform_size):
    """
    加载图片并进行预处理
    """
    img = Image.open(url)
    img = np.array(img)[:, :, :3]
    H, W = img.shape[0], img.shape[1]
    img = Image.fromarray(img)
    rgb_transform = transforms.Compose(
                [
                    transforms.Resize((transform_size, transform_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
    img = rgb_transform(img).to('cuda')
    img = img.unsqueeze(0).detach()
    return img, H, W

class CLIPFeatureExtractor(nn.Module):
    """
    修改的CLIP模型，用于提取中间层特征图
    支持更大的输入分辨率以获得更高分辨率的特征图
    """
    def __init__(self, layer=11, input_resolution=1600):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device='cuda')
        self.layer = layer
        self.input_resolution = input_resolution
        
        # 计算patch grid size
        self.patch_size = 32  # ViT-B/32的patch size
        self.grid_size = input_resolution // self.patch_size
        
    def interpolate_pos_embed(self, pos_embed, grid_size):
        """
        插值位置编码以适应不同的输入分辨率
        """
        # pos_embed shape: [1 + 7*7, 768] for ViT-B/32
        # 第一个是class token，后面49个是7x7的patch位置编码
        
        cls_pos_embed = pos_embed[:1, :]  # class token的位置编码
        patch_pos_embed = pos_embed[1:, :]  # patch的位置编码
        
        # 原始的patch grid是7x7
        orig_grid_size = 7
        patch_pos_embed = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, -1)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, 768, 7, 7]
        
        # 插值到新的grid size
        patch_pos_embed = F.interpolate(
            patch_pos_embed, 
            size=(grid_size, grid_size), 
            mode='bicubic', 
            align_corners=False
        )
        
        # 重新整理形状
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # [1, grid_size, grid_size, 768]
        patch_pos_embed = patch_pos_embed.reshape(grid_size * grid_size, -1)
        
        # 合并class token和patch位置编码
        new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=0)
        
        return new_pos_embed
        
    def forward(self, x):
        visual = self.model.visual
        
        # 确保数据类型匹配
        x = x.to(visual.conv1.weight.dtype)
        
        # Patch embedding
        x = visual.conv1(x)  # shape: [*, width, grid, grid]
        grid_size = x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape: [*, grid ** 2, width]
        
        # 添加class token
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        
        # 插值位置编码以适应新的grid size
        if grid_size != 7:  # 如果不是标准的7x7 grid
            pos_embed = self.interpolate_pos_embed(visual.positional_embedding, grid_size)
            x = x + pos_embed.to(x.dtype)
        else:
            x = x + visual.positional_embedding.to(x.dtype)
            
        x = visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 通过指定层数的transformer blocks
        for i in range(self.layer + 1):
            x = visual.transformer.resblocks[i](x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 移除class token，只保留patch tokens
        patch_tokens = x[:, 1:, :]  # 移除第一个token (class token)
        
        # 重新整理为特征图格式 [batch, feature_dim, height, width]
        feature_map = patch_tokens.transpose(1, 2).reshape(patch_tokens.shape[0], patch_tokens.shape[2], grid_size, grid_size)
        
        return feature_map

# 5. 特征向量可视化
def clip_feature_visualization(img_path, transform_size=3200):
    """
    使用CLIP对图片进行特征提取和PCA可视化
    """
    print(f"开始处理图片: {img_path}")
    
    # 加载图片
    img, H, W = load_image(img_path, transform_size)
    print(f"原始图片尺寸: {W} x {H}")
    print(f"预处理后张量形状: {img.shape}")
    
    # 创建特征提取器
    model = CLIPFeatureExtractor(layer=11, input_resolution=transform_size).to('cuda')
    model.eval()
    
    with torch.no_grad():
        # 提取特征图
        feature_map = model(img)
        print(f"特征图形状: {feature_map.shape}")
        
        # PCA可视化
        pca_result = PCA_visualize(feature_map, H, W, return_res=True)
        
        # 显示结果
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示原始图片
        original_img = Image.open(img_path)
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 显示PCA可视化结果
        axes[1].imshow(pca_result)
        axes[1].set_title(f'​​CLIP Feature PCA Visualization​\nPCA Dimensionality Reduction Output Dimensions​: {pca_result.shape}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('clip_feature_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_map, pca_result

# 执行特征可视化
if __name__ == "__main__":
    img_path = '/home/lumina/lumina/Jiawei/extracted_images/Extracted_img_front_camera_random/right_camera_frame000.jpg'
    if os.path.exists(img_path):
        print(f"开始处理图片: {img_path}")
        feature_map, visualization = clip_feature_visualization(img_path)
        print("特征可视化完成！")
    else:
        print(f"错误: 文件 {img_path} 不存在！")
        print("可用的图片文件:")
        if os.path.exists('extracted_images'):
            for f in sorted(os.listdir('extracted_images'))[:5]:  # 显示前5个文件
                print(f"  {f}")


# # 4. 将jpg图片输入到clip的image encoder得到特征图
# def jpg_2_feature(img_path):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, transform = clip.load("ViT-B/32",device=device)
#     model.eval()
#     with Image.open(img_path) as img:
#         img = transform(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             feature = model.encode_image(img)
#     return feature
# feature = jpg_2_feature('extracted_images/front_camera_frame000.jpg')
# print(feature.shape)

# # 3. 查看jpg图片的内容
# # img_path = 'extracted_images/front_camera_frame000.jpg'
# img_path = 'clip_feature_visualization.png'
# if os.path.exists(img_path):
#     with Image.open(img_path) as img:
#         print("===== 图像信息 =====")
#         print(f"格式: {img.format}")
#         print(f"模式: {img.mode}")  # 如 'RGB', 'L' (灰度)
#         print(f"分辨率: {img.size}")  # (宽度, 高度)
#         print(f"元数据: {img.info}")  # 如 EXIF 数据
#         # 转换为NumPy数组并获取维度
#         img_array = np.array(img)
#         print(f"数组形状 (高×宽×通道): {img_array.shape}")
# else:
#     print(f"错误: 文件 {img_path} 不存在！")

# # 2. 把hdf5转换为jpg
# file_path = 'RoboTwin/data/move_can_pot/demo_clean/data/episode0.hdf5'
# output_dir = 'extracted_images'
# os.makedirs(output_dir, exist_ok=True)
# with h5py.File(file_path, 'r') as f:
#     for cam_name in ['front_camera', 'head_camera', 'left_camera', 'right_camera']:
#         rgb_data = f[f'observation/{cam_name}/rgb'][:]  # 获取二进制字符串
#         for i, byte_str in enumerate(rgb_data):
#             img = Image.open(io.BytesIO(byte_str))
#             img.save(f'{output_dir}/{cam_name}_frame{i:03d}.jpg')

# # 1. 查看hdf5内容和结构
# file_path = 'RoboTwin/data/move_can_pot/demo_clean/data/episode0.hdf5'
# def explore_hdf5(file_path):
#     with h5py.File(file_path, 'r') as f:
#         print("===== HDF5 文件结构 =====")
#         print(f"文件根目录下的键: {list(f.keys())}\n")
#         def print_attrs(name, obj):
#             print(f"数据集/组名: {name}")
#             if isinstance(obj, h5py.Dataset):
#                 print(f"  形状: {obj.shape}")
#                 print(f"  数据类型: {obj.dtype}")
#             if obj.attrs:
#                 print(f"  属性: {dict(obj.attrs)}")
#             print("------")
#         f.visititems(print_attrs)
# explore_hdf5(file_path)

