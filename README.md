# Deep Learning Change Detection (geoskill-change-detection-dl)

> Siamese fully convolutional network (FC-Siam-diff style) for bi-temporal change detection: GPU training/inference, outputting change probability, a binary change map and change regions

---

## 1. Overview

Detects surface changes from bi-temporal imagery (vegetation degradation, urban expansion, water body growth/shrinkage), outputting a change probability map, a binary change map and change regions as GeoJSON. This skill uses a real deep learning model: a **Siamese fully convolutional change detection network** (FC-Siam-diff style, Daudt et al. 2018). The two timestamps (red/nir dual channels) are fed through a shared-weight encoder to extract multi-scale features; the decoder fuses per-level feature differences |f1 − f2| to reconstruct pixel-level change probability (sigmoid, [0,1]), which is binarized with the `--prob-thresh` threshold, aggregated into change regions via 8-neighborhood connected components, and georeferenced. Both training and inference run on CUDA GPUs (torch ≥ 2.x, requiring cuDNN or automatically falling back to native CUDA convolutions). Pre-trained weights `cd_siamese_weights.pt` are bundled with the skill (≈0.6 MB, trained on synthetic bi-temporal change pairs with holdout recall ≈ 0.999 / false-alarm ≈ 0). If the weight file is missing, the model is automatically trained on synthetic data on the GPU at first run (≈15 seconds) and cached to disk.

## 3. Quick Start

```bash
pip install -r requirements.txt
python geoskill-change-detection-dl.py --bbox 116 39 117 40 --synthetic --output-dir ./out
```

## 4. CLI Parameters

Run `python <skill>.py --help` for the full list. Common parameters:

| Parameter | Type | Description |
|---|---|---|
| `--bbox` | `float[4]` | WGS84 bounding box `min_lon min_lat max_lon max_lat` |
| `--input` | `path` | Local input file (GeoJSON/GeoTIFF/etc.) |
| `--output-dir` | `path` | Output directory (default `./output`) |
| `--synthetic` | `flag` | Use synthetic data instead of real input |
| `--quiet` | `flag` | Suppress non-essential stdout |

## 5. Input / Output

| File | Format | Description |
|---|---|---|
| `change_probability.tif` | GeoTIFF | Change probability [0,1] (network sigmoid output) |
| `change_binary.tif` | GeoTIFF | Binary change map |
| `change_regions.geojson` | GeoJSON | Change region polygons + area attributes |
| `output-manifest.json` | JSON | Run manifest (input/output/QA/exit code/model metadata) |

In synthetic mode, QA writes `synthetic_recall` / `synthetic_false_alarm` (computed against the built-in ground truth).


## 6. Technical Principle

(see SKILL.md for details)

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

> 中文完整版：[references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)
