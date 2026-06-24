# Garden 3DGS Reconstruction, Object Insertion, Rendering



```bash
git clone https://github.com/groolegend/3D_garden_reconstruction.git
cd 3D_garden_reconstruction
```


目录结构：

```text
3DGS/
├── data/
│   └── mipnerf360/
│       └── garden/                 # 下载后的 Mip-NeRF 360 garden 数据
├── 3D_data/
│   ├── objB/model.obj              # 待插入物体之一
│   ├── objC/model.obj              # 待插入物体之一
│   └── point_cloud.ply             # 第三个待插入物体
├── output/
│   └── mipnerf360/
│       ├── garden_wandb_v2/        # 训练得到的 garden 3DGS 模型
│       └── garden_on_table/        # 插入物体后的输出
└── scripts/
```

## 1. 环境创建

本项目使用 `uv` 创建 Python 环境。系统中已安装 `uv` 时，可直接运行：

```bash
bash scripts/setup_uv_env.sh
source .venv/bin/activate
```

该脚本会完成以下工作：

- 创建 `.venv`
- 安装 PyTorch CUDA 版本
- 安装 `plyfile`、`tqdm`、`opencv-python`、`joblib`
- 编译并安装 3DGS 依赖：
  - `diff-gaussian-rasterization`
  - `simple-knn`
  - `fused-ssim`


## 2. 下载 Garden 数据

只下载 Mip-NeRF 360 中的 `garden` 场景：

```bash
source .venv/bin/activate

bash scripts/download_mipnerf360.sh data/mipnerf360 garden
```

下载完成后，数据目录应为：

```text
data/mipnerf360/garden
data/mipnerf360/garden/images_4
data/mipnerf360/garden/sparse/0
```

本实验使用 `images_4` 作为训练图像目录，以降低显存和训练时间。

## 3. 训练 Garden 背景模型

### 3.1 普通训练

可以使用项目脚本直接完成训练、渲染和指标计算：

```bash
source .venv/bin/activate

bash scripts/reconstruct_garden.sh \
  data/mipnerf360 \
  output/mipnerf360
```

训练结果输出到：

```text
output/mipnerf360/garden
```

最终 3DGS 模型参数保存在：

```text
output/mipnerf360/garden/point_cloud/iteration_30000/point_cloud.ply
```

在 3DGS 中，这个 `point_cloud.ply` 就是主要的“模型权重”。它保存了每个 Gaussian 的位置、颜色、opacity、scale、rotation 等可学习参数。

### 3.2 使用 W&B 记录训练曲线

先登录 W&B：

```bash
wandb login
```

然后训练：

```bash
source .venv/bin/activate

python train.py \
  -s data/mipnerf360/garden \
  -i images_4 \
  -m output/mipnerf360/garden_wandb_v2 \
  --eval \
  --wandb \
  --wandb_project 3dgs-garden \
  --wandb_name garden_wandb_v2 \
  --test_iterations 1000 3000 7000 15000 30000 \
  --save_iterations 7000 30000 \
  --disable_viewer
```

本实验使用上述命令训练得到的结果：

```text
ITER 30000
test  L1   = 0.0270
test  PSNR = 27.50
train L1   = 0.0177
train PSNR = 31.55
```

训练曲线中每隔 `3000` iter 出现一次 loss 尖峰是正常现象，主要由 `opacity_reset_interval=3000` 触发的 opacity reset 导致。尖峰后 loss 会快速回落。

## 4. 加载已有模型并测试

### 4.1 下载已有权重

可以下载已经训练好的 3DGS 权重。

纯 garden 背景权重：

[Download garden 3DGS weights](https://drive.google.com/file/d/1MQMnZhDAauuN_cvvPBFSMN79oI1BJUYo/view?usp=sharing)

下载后解压或放置为以下目录结构：

```text
output/mipnerf360/garden_wandb_v2/
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply
├── cameras.json
├── cfg_args
├── exposure.json
└── input.ply
```

加入三个物体后的 garden 权重：

[Download garden with inserted objects weights](https://drive.google.com/file/d/1kGytVkJWQ8qxFKWvYzRAKjIjZW0LOV5i/view?usp=sharing)

下载后解压或放置为以下目录结构：

```text
output/mipnerf360/garden_on_table/
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply
├── cameras.json
├── cfg_args
├── exposure.json
└── input.ply
```

### 4.2 测试纯 Garden 背景模型

如果已经有训练好的模型，不需要重新训练，可以直接渲染测试集：

```bash
source .venv/bin/activate

python render.py \
  -s data/mipnerf360/garden \
  -i images_4 \
  -m output/mipnerf360/garden_wandb_v2 \
  --iteration 30000 \
  --eval
```

使用上面的 `render.py` 命令后，渲染结果写入：

```text
output/mipnerf360/garden_wandb_v2/test/ours_30000/renders
output/mipnerf360/garden_wandb_v2/test/ours_30000/gt
output/mipnerf360/garden_wandb_v2/train/ours_30000/renders
```

计算 PSNR / SSIM / LPIPS：

```bash
python metrics.py \
  -m output/mipnerf360/garden_wandb_v2
```

### 4.3 渲染加入物体后的 Garden 模型

如果已经下载并放置了加入物体后的权重，可以直接渲染：

```bash
source .venv/bin/activate

python render.py \
  -s data/mipnerf360/garden \
  -i images_4 \
  -m output/mipnerf360/garden_on_table \
  --iteration 30000 \
  --skip_test \
  --quiet \
  -r 960
```

使用上面的命令后，渲染帧写入：

```text
output/mipnerf360/garden_on_table/train/ours_30000/renders
```

合成视频：

```bash
ffmpeg -y \
  -framerate 24 \
  -i output/mipnerf360/garden_on_table/train/ours_30000/renders/%05d.png \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
  -c:v libx264 \
  -crf 18 \
  output/mipnerf360/garden_on_table/garden_with_objects_walkthrough.mp4
```

## 5. 准备三个物体

三个待插入物体放在：

```text
3D_data
```

该目录同样不随 GitHub 仓库发布，需要用户自行准备。放置完成后的目录为：

`scripts/insert_garden_objects.py` 读取以下文件：

```text
3D_data/objB/model.obj
3D_data/objC/model.obj
3D_data/point_cloud.ply
```

其中：

- `objB/model.obj`：OBJ mesh 物体，脚本会从表面采样点并读取贴图颜色。
- `objC/model.obj`：OBJ mesh 物体，脚本会从表面采样点并读取贴图颜色。
- `point_cloud.ply`：已经是 Gaussian/点云形式的物体，脚本会直接缩放并放置。

如果三个物体的文件名不同，需要修改 `scripts/insert_garden_objects.py` 中的 `placements` 和 `third_ply` 路径。

## 6. 把三个物体放到 Garden 桌子上

本项目使用代码级拼接方式完成融合：

1. 读取训练好的 garden `point_cloud.ply`
2. 从 garden 高斯点云中估计桌面平面
3. 将 OBJ mesh 物体采样为带颜色点云
4. 将物体点云转换为 Gaussian 参数
5. 根据桌面平面，把物体底部对齐到桌面上方
6. 合并背景 Gaussian 和物体 Gaussian
7. 调用 3DGS renderer 渲染图像并用 ffmpeg 合成视频

运行命令：

```bash
source .venv/bin/activate

python scripts/insert_garden_objects.py \
  --model output/mipnerf360/garden_wandb_v2 \
  --source data/mipnerf360/garden \
  --objects 3D_data \
  --output output/mipnerf360/garden_on_table \
  --iteration 30000 \
  --resolution 960 \
  --fps 24
```

输出模型：

```text
output/mipnerf360/garden_on_table/point_cloud/iteration_30000/point_cloud.ply
```

渲染帧：

```text
output/mipnerf360/garden_on_table/train/ours_30000/renders
```

最终视频：

```text
output/mipnerf360/garden_on_table/garden_with_objects_walkthrough.mp4
```

如果只想生成合并后的 Gaussian 模型，暂时不渲染视频：

```bash
python scripts/insert_garden_objects.py \
  --model output/mipnerf360/garden_wandb_v2 \
  --source data/mipnerf360/garden \
  --objects 3D_data \
  --output output/mipnerf360/garden_on_table \
  --iteration 30000 \
  --skip-render
```

## 7. 灰色物体的可视化选项

如果某个 OBJ 模型纹理本身是纯灰色，直接渲染时可能缺少明暗变化，看起来像灰色块。脚本提供了 `objB` 的伪 shading 选项：

```bash
# 保持原始灰色纹理
--objb-shading none

# 根据表面法线添加轻微明暗
--objb-shading normal

# 添加更明显的伪纹理变化
--objb-shading fake_texture
```

例如：

```bash
python scripts/insert_garden_objects.py \
  --model output/mipnerf360/garden_wandb_v2 \
  --source data/mipnerf360/garden \
  --objects 3D_data \
  --output output/mipnerf360/garden_on_table_fake_texture \
  --iteration 30000 \
  --resolution 960 \
  --fps 24 \
  --objb-shading fake_texture
```

## 8. 单独渲染插入后的模型

如果已经生成了 `garden_on_table`，也可以单独重新渲染：

```bash
source .venv/bin/activate

python render.py \
  -s data/mipnerf360/garden \
  -i images_4 \
  -m output/mipnerf360/garden_on_table \
  --iteration 30000 \
  --skip_test \
  --quiet \
  -r 960
```

然后使用 ffmpeg 合成视频：

```bash
ffmpeg -y \
  -framerate 24 \
  -i output/mipnerf360/garden_on_table/train/ours_30000/renders/%05d.png \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
  -c:v libx264 \
  -crf 18 \
  output/mipnerf360/garden_on_table/garden_with_objects_walkthrough.mp4
```



## 9. 完整复现实验命令

从零开始的最小流程：

```bash
cd 3DGS

# 1. 创建环境
bash scripts/setup_uv_env.sh
source .venv/bin/activate

# 2. 下载 garden
bash scripts/download_mipnerf360.sh data/mipnerf360 garden

# 3. 训练 garden 背景
python train.py \
  -s data/mipnerf360/garden \
  -i images_4 \
  -m output/mipnerf360/garden_wandb_v2 \
  --eval \
  --wandb \
  --wandb_project 3dgs-garden \
  --wandb_name garden_wandb_v2 \
  --test_iterations 1000 3000 7000 15000 30000 \
  --save_iterations 7000 30000 \
  --disable_viewer

# 4. 测试指标
python render.py \
  -s data/mipnerf360/garden \
  -i images_4 \
  -m output/mipnerf360/garden_wandb_v2 \
  --iteration 30000 \
  --eval

python metrics.py \
  -m output/mipnerf360/garden_wandb_v2

# 5. 插入三个物体并生成视频
python scripts/insert_garden_objects.py \
  --model output/mipnerf360/garden_wandb_v2 \
  --source data/mipnerf360/garden \
  --objects 3D_data \
  --output output/mipnerf360/garden_on_table \
  --iteration 30000 \
  --resolution 960 \
  --fps 24
```
