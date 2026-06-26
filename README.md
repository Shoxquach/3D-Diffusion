# 点云三维扩散模型（ModelNet40 单类无条件生成）

基于 DDPM 的点云扩散模型，使用 ModelNet40 网格数据训练，从随机噪声去噪生成三维点云。

## 环境

- Python 3.10+
- PyTorch 2.0+（建议 CUDA）
- 依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 数据

数据放在 `data/ModelNet40/<category>/train|test/*.off`。

默认类别：`airplane`。首次运行会自动将网格采样为点云并缓存到 `data/ModelNet40/_cache/`。

## 快速测试

```bash
python scripts/test_pipeline.py
```

## 训练

```bash
python train.py --config configs/airplane.yaml
```

常用参数覆盖：

```bash
python train.py --config configs/airplane.yaml --epochs 100 --batch_size 16 --device cuda
```

检查点保存在 `checkpoints/`。

## 采样生成

```bash
python sample.py --checkpoint checkpoints/latest.pt --num_samples 8 --steps 50 --method ddim
```

输出保存在 `outputs/samples/`，包含 `.npy` 和 `.ply` 文件。可用 MeshLab / CloudCompare / Open3D 查看 PLY。

## 项目结构

```
├── configs/airplane.yaml   # 训练配置
├── data/dataset.py         # 数据集
├── models/
│   ├── diffusion.py        # DDPM 加噪/去噪
│   ├── denoiser.py         # Transformer 去噪网络
│   └── point_e/            # Point-E 主干实现
├── utils/mesh.py           # 网格采样与归一化
├── train.py
├── sample.py
└── scripts/test_pipeline.py
```

## 说明

- 去噪网络为 **Point-E 风格 Transformer**（默认 256 维 / 6 层，约 5M 参数）。
- 训练损失为预测噪声 MSE；推理默认 DDIM 50 步加速。
- 完整训练建议 200 epoch 以上，并观察生成点云质量再调参。
