# 深度学习变化检测 | Deep Learning Change Detection

从双时相影像检测地表变化（植被退化、城市扩张、水体消长），输出变化概率图、二值变化图与变化图斑 GeoJSON。

本 skill 使用真正的深度学习模型：**Siamese 全卷积变化检测网络**（FC-Siam-diff 风格，
Daudt et al. 2018）。两个时相（red/nir 双通道）经共享权重编码器提取多尺度特征，
解码器融合逐层特征差 |f1 − f2| 重建像元级变化概率（sigmoid，[0,1]），
再经 `--prob-thresh` 阈值二值化、8 邻域连通域聚合为变化图斑并地理编码。
训练与推理均在 CUDA GPU 上执行（torch ≥ 2.x，需 cuDNN 或自动退回 CUDA 原生卷积）。

随 skill 附带预训练权重 `cd_siamese_weights.pt`（约 0.6 MB，在合成双时相变化对上
训练，holdout recall ≈ 0.999 / false-alarm ≈ 0）。若权重文件缺失，首次运行时
自动在 GPU 上用合成数据训练（约 15 秒）并落盘缓存。

## 依赖

```bash
pip install numpy rasterio scipy torch
```

## 使用方法

### 示例 1（合成数据，离线）

```bash
python geoskill-change-detection-dl.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### 示例 2：双文件双时相（各 2 波段 [red, nir] 反射率）

```bash
python geoskill-change-detection-dl.py --input t1.tif --input2 t2.tif --prob-thresh 0.6 --output-dir ./out
```

### 示例 3：单文件 4 波段 [red1,nir1,red2,nir2]

```bash
python geoskill-change-detection-dl.py --input pair.tif --min-area 16 --output-dir ./out
```

## 输出

| 文件 | 格式 | 说明 |
|---|---|---|
| `change_probability.tif` | GeoTIFF | 变化概率 [0,1]（网络 sigmoid 输出） |
| `change_binary.tif` | GeoTIFF | 二值变化图 |
| `change_regions.geojson` | GeoJSON | 变化图斑多边形 + 面积属性 |
| `output-manifest.json` | JSON | 运行清单（输入/输出/QA/退出码/模型元信息） |

合成模式 QA 会写入 `synthetic_recall` / `synthetic_false_alarm`（对内置真值计算）。

## 局限（诚实声明）

- 随附权重在合成光谱（植被/裸土/水体/建成区四类反射率）上训练，对真实影像的
  概率输出未做辐射定标/外场验证；真实数据结果应视为筛查级而非制图级。
- `--scale` 为旧版经典基线（1−exp(−scale·|dNDVI|)）的陡峭度参数，保留仅为
  CLI 兼容，网络概率路径不使用它。

## 数据源 / Source

本地双时相 GeoTIFF（各含 red/nir 波段，或单文件 4 波段），或 --synthetic 合成对
（t1 全植被，t2 中心退化为裸土）。

## 隐私声明 / Privacy

- 默认离线运行，`--synthetic` 模式完全无网络。
- 所有处理在本地完成，不上传用户数据。

## License

MIT
