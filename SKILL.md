---
name: geoskill-change-detection-dl
description: 'Siamese 全卷积网络(FC-Siam-diff 风格)双时相变化检测：GPU 训练/推理，输出变化概率、二值变化图与变化图斑'
---

# 深度学习变化检测 | Deep Learning Change Detection

Detects land surface changes from bi-temporal imagery (vegetation degradation, urban expansion, water body growth/shrinkage), outputting a change probability map, a binary change map, and change-region polygons as GeoJSON.

This skill uses a genuine deep learning model: a **Siamese fully convolutional change detection network** (FC-Siam-diff style, Daudt et al. 2018). The two epochs (red/nir two-channel) pass through a shared-weight encoder to extract multi-scale features, and the decoder fuses layer-wise feature differences |f1 − f2| to reconstruct pixel-level change probabilities (sigmoid, [0, 1]), which are then thresholded via `--prob-thresh`, aggregated into change regions through 8-neighborhood connected-component analysis, and geocoded. Both training and inference run on CUDA GPUs (torch ≥ 2.x, requiring cuDNN or automatically falling back to native CUDA convolutions).

The skill ships with pretrained weights `cd_siamese_weights.pt` (about 0.6 MB, trained on synthetic bi-temporal change pairs, holdout recall ≈ 0.999 / false alarm ≈ 0). If the weight file is missing, the first run automatically trains on synthetic data on the GPU (about 15 seconds) and caches the weights to disk.

## Dependencies / 依赖

```bash
pip install numpy rasterio scipy torch
```

## Usage / 使用方法

### Example 1 (Synthetic Data, Offline)

```bash
python geoskill-change-detection-dl.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### Example 2: Two Files for Bi-temporal Input (Each with 2 Bands [red, nir] Reflectance)

```bash
python geoskill-change-detection-dl.py --input t1.tif --input2 t2.tif --prob-thresh 0.6 --output-dir ./out
```

### Example 3: Single File with 4 Bands [red1, nir1, red2, nir2]

```bash
python geoskill-change-detection-dl.py --input pair.tif --min-area 16 --output-dir ./out
```

## Output / 输出

| File | Format | Description |
|---|---|---|
| `change_probability.tif` | GeoTIFF | Change probability [0, 1] (network sigmoid output) |
| `change_binary.tif` | GeoTIFF | Binary change map |
| `change_regions.geojson` | GeoJSON | Change-region polygons + area attributes |
| `output-manifest.json` | JSON | Run manifest (inputs/outputs/QA/exit code/model metadata) |

In synthetic mode, QA writes `synthetic_recall` / `synthetic_false_alarm` (computed against the built-in ground truth).

## Limitations / 局限（诚实声明）

- The bundled weights were trained on synthetic spectra (four reflectance classes: vegetation/bare soil/water/built-up); the probability outputs on real imagery have not been radiometrically calibrated or field-validated, so results on real data should be treated as screening-level rather than map-grade.
- `--scale` is the steepness parameter of the legacy classical baseline (1−exp(−scale·|dNDVI|)); it is kept only for CLI compatibility and is not used by the network probability path.

## Data Source / 数据源 / Source

Local bi-temporal GeoTIFFs (each containing red/nir bands, or a single 4-band file), or a `--synthetic` pair (t1 fully vegetated, t2 degraded to bare soil at the center).

## Privacy / 隐私声明 / Privacy

- Runs offline by default; `--synthetic` mode requires no network access at all.
- All processing is performed locally; no user data is uploaded.

## License / License

MIT

---

> 中文完整版：需要时再读 [references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)，不要默认全量加载。
