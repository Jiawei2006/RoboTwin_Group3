import h5py

file_path = "/home/lumina/lumina/Jiawei/RoboTwin/data/move_can_pot/demo_clean/data/episode0.hdf5"
with h5py.File(file_path, 'r') as f:
    def print_attrs(name, obj):
        print(f"📂 路径: {name}")
        if isinstance(obj, h5py.Dataset):
            print(f"   ├── 类型: Dataset")
            print(f"   ├── 数据类型: {obj.dtype}")
            print(f"   ├── 维度: {obj.shape}")
        elif isinstance(obj, h5py.Group):
            print(f"   ├── 类型: Group")
        if obj.attrs:
            print(f"   ├── 属性: {dict(obj.attrs)}")
        print("")

    print("===== HDF5 文件结构如下 =====")
    f.visititems(print_attrs)

'''
/RoboTwin/data/move_can_pot/demo_clean/data/episode0.hdf5
├── endpose/                         # 组（目前为空）
│
├── joint_action/                   # 组：机器人双臂动作
│   ├── left_arm            (154, 6)      float64
│   ├── left_gripper        (154,)        float64
│   ├── right_arm           (154, 6)      float64
│   ├── right_gripper       (154,)        float64
│   └── vector              (154, 14)     float64
│
├── observation/                    # 组：多相机观测数据
│   ├── front_camera/
│   │   ├── cam2world_gl     (154, 4, 4)   float32
│   │   ├── extrinsic_cv     (154, 3, 4)   float32
│   │   ├── intrinsic_cv     (154, 3, 3)   float32
│   │   └── rgb              (154,)        bytes (S14990)
│   │
│   ├── head_camera/
│   │   ├── cam2world_gl     (154, 4, 4)   float32
│   │   ├── extrinsic_cv     (154, 3, 4)   float32
│   │   ├── intrinsic_cv     (154, 3, 3)   float32
│   │   └── rgb              (154,)        bytes (S15333)
│   │
│   ├── left_camera/
│   │   ├── cam2world_gl     (154, 4, 4)   float32
│   │   ├── extrinsic_cv     (154, 3, 4)   float32
│   │   ├── intrinsic_cv     (154, 3, 3)   float32
│   │   └── rgb              (154,)        bytes (S6038)
│   │
│   └── right_camera/
│       ├── cam2world_gl     (154, 4, 4)   float32
│       ├── extrinsic_cv     (154, 3, 4)   float32
│       ├── intrinsic_cv     (154, 3, 3)   float32
│       └── rgb              (154,)        bytes (S15510)
│
└── pointcloud              (154, 0)       float64     # 空点云数据
'''
