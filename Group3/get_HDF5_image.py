import h5py
import os
import io
from PIL import Image
from tqdm import tqdm

# 路径配置
hdf5_path = "/home/lumina/lumina/Jiawei/RoboTwin/data/move_can_pot/demo_clean/data/episode0.hdf5"
output_dir = "/home/lumina/lumina/Jiawei/Extracted_img_front_camera_random"
os.makedirs(output_dir, exist_ok=True)

# 打开 HDF5 文件
with h5py.File(hdf5_path, 'r') as f:
    # 读取字节形式的图像序列
    rgb_data = f["observation/front_camera/rgb"][:]  # shape: (154,)
    
    print(f"共提取 {len(rgb_data)} 帧图像，正在保存至: {output_dir}")
    
    for i, byte_str in tqdm(enumerate(rgb_data), total=len(rgb_data)):
        try:
            img = Image.open(io.BytesIO(byte_str))  # 解码为图像
            save_path = os.path.join(output_dir, f"right_camera_frame{i:03d}.jpg")
            img.save(save_path)
        except Exception as e:
            print(f"[警告] 第 {i} 帧解码失败: {e}")
