#!/usr/bin/env python3
"""change-detection-dl — 深度学习变化检测

从双时相遥感影像中检测地表变化（如植被退化、城市扩张、水体消长），
输出变化概率图、二值变化图与变化图斑 GeoJSON。

**核心模型**：Siamese 全卷积变化检测网络（FC-Siam-diff 风格，
Daudt et al. 2018, ICIP）：两个时相共享权重编码器提取多尺度特征，
解码器融合两时相特征的逐层绝对差 |f1 - f2| 重建变化概率图，
sigmoid 输出 [0, 1] 像元级变化概率。训练/推理均在 CUDA GPU 上执行
（torch >= 2.x）。随 skill 附带预训练权重 ``cd_siamese_weights.pt``
（在合成双时相变化对上训练）；若权重缺失则在首次运行时自动用 GPU
训练并落盘缓存。

数据流：
1. 加载双时相 red/nir 反射率（双文件各 2 波段，或单文件 4 波段
   [red1, nir1, red2, nir2]）；
2. Siamese FCN 前向 → 变化概率 [0, 1]；
3. ``--prob-thresh`` 阈值二值化；
4. 8 邻域连通域聚合为变化图斑，地理编码输出 GeoJSON。

数据源：本地双时相 GeoTIFF，或 ``--synthetic`` 生成含一处人为变化区
的模拟对（合成模式用保留真值计算 recall / false-alarm 写入 manifest QA）。

隐私声明 / Privacy：
- 默认离线运行，``--synthetic`` 模式完全无网络。
- 所有处理在本地完成，不上传用户数据。

Usage:
    python change-detection-dl.py --input t1.tif --input2 t2.tif --output-dir ./out
    python change-detection-dl.py --bbox 116 39 117 40 --synthetic --output-dir ./out

License: MIT
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
SKILL_NAME = "change-detection-dl"

try:
    from _geoskill_core.errors import (
        GeoSkillError, UsageError, DependencyError, ValidationError, ProcessError,
        to_exit_code,
    )
    from _geoskill_core.manifest import OutputManifest, OutputFile
except ImportError:  # pragma: no cover
    class GeoSkillError(Exception):
        def __init__(self, message: str, code: int = 7, kind: str = "EGeo", **kw):
            super().__init__(message)
            self.message, self.code, self.kind = message, code, kind

    class UsageError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=2, kind="EUsage", **k)

    class DependencyError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=3, kind="EDep", **k)

    class ValidationError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=6, kind="EValidate", **k)

    class ProcessError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=7, kind="EProcess", **k)

    def to_exit_code(exc):
        return getattr(exc, "code", 7)

    OutputManifest = None
    OutputFile = None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# 经典算子（保留：单元测试覆盖 + QA 统计用）
# ---------------------------------------------------------------------------
def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算 NDVI = (NIR - Red) / (NIR + Red)，范围 [-1, 1]。"""
    red = np.asarray(red, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    if red.shape != nir.shape:
        raise ValidationError("red and nir shape mismatch",
                              red=list(red.shape), nir=list(nir.shape))
    denom = nir + red
    out = np.divide(nir - red, denom, out=np.zeros_like(nir), where=np.abs(denom) > 1e-9)
    return np.clip(out, -1.0, 1.0)


def change_difference(ndvi_t1: np.ndarray, ndvi_t2: np.ndarray) -> np.ndarray:
    """时相 NDVI 差分：dNDVI = t2 - t1（QA/诊断用）。"""
    a = np.asarray(ndvi_t1, dtype=np.float64)
    b = np.asarray(ndvi_t2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValidationError("ndvi shape mismatch", t1=list(a.shape), t2=list(b.shape))
    return b - a


def change_probability(diff: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """经典基线：把 |dNDVI| 经指数饱和映射到 [0, 1]（1 - exp(-scale*|d|)）。

    注：主推理路径使用 Siamese 网络学习得到的概率；本函数为可验证的
    经典等价映射，保留用于单元测试与对照。
    """
    d = np.abs(np.asarray(diff, dtype=np.float64))
    prob = 1.0 - np.exp(-scale * d)
    return np.clip(prob, 0.0, 1.0)


def binary_change(prob: np.ndarray, prob_thresh: float = 0.5) -> np.ndarray:
    """概率阈值化得到二值变化图 (bool)。"""
    return np.asarray(prob, dtype=np.float64) >= prob_thresh


def change_regions(
    binary: np.ndarray,
    min_area: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """连通域聚合变化像元为图斑，返回 (label_map, regions)。"""
    from scipy.ndimage import label
    binary = np.asarray(binary).astype(bool)
    labels, n = label(binary, structure=np.ones((3, 3), dtype=bool))
    regions: List[Dict[str, Any]] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        area = int(ys.size)
        if area < min_area:
            continue
        regions.append({
            "region_id": int(lab),
            "area_px": area,
            "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid_row": float(np.mean(ys)),
            "centroid_col": float(np.mean(xs)),
        })
    regions.sort(key=lambda d: d["area_px"], reverse=True)
    return labels.astype(np.int32), regions


# ---------------------------------------------------------------------------
# 深度学习核心：Siamese FCN（FC-Siam-diff 风格）变化检测网络
# 参考: Daudt, Le Saux, Boulch, "Fully convolutional siamese networks for
#       change detection", ICIP 2018, pp. 4063-4067 (arXiv:1810.08462)
# ---------------------------------------------------------------------------
WEIGHTS_FILENAME = "cd_siamese_weights.pt"
_MODEL_CACHE: Dict[str, Any] = {}

# 训练用合成地物光谱（red, nir 反射率中心值），与 generate_synthetic 一致
_CLASSES = ("veg", "soil", "water", "built")
_SPECTRA = {
    "veg":   (0.050, 0.50),   # 健康植被：低红高近红外
    "soil":  (0.260, 0.29),   # 裸土：红/近红外接近且偏高
    "water": (0.050, 0.08),   # 水体：近红外强吸收
    "built": (0.190, 0.36),   # 建成区：中等红、中等近红外
}


def _prepare_dll_paths() -> None:
    """Windows 下把 conda 环境的 Library\\bin 注册进 DLL 搜索路径。

    torch 的 cuDNN 依赖（cudnn64_9.dll / cudnn_graph64_9.dll / cudart64_12.dll
    等）装在 <env>\\Library\\bin；若进程不是从激活的 conda 环境启动，
    PATH 里没有该目录，torch 加载 cuDNN 会失败。见 SHARED_ISSUES ENV-003。
    """
    if os.name != "nt":
        return
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if not os.path.isdir(lib_bin):
        return
    if lib_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(lib_bin)
    except (OSError, AttributeError):
        pass


def _require_torch():
    _prepare_dll_paths()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise DependencyError(
            "torch is required for change-detection-dl (pip install torch)") from exc
    import torch
    return torch


def _cuda_device(torch_mod):
    if not torch_mod.cuda.is_available():
        raise DependencyError(
            "change-detection-dl requires a CUDA GPU; torch.cuda.is_available() "
            "is False in this environment")
    # cuDNN 探测：DLL 路径注册成功则启用（加速卷积）；探测抛异常时关闭 cuDNN，
    # 卷积退回 CUDA 原生实现（仍在 GPU，不降级 CPU）。
    try:
        torch_mod.backends.cudnn.enabled = bool(torch_mod.backends.cudnn.is_available())
    except Exception:  # pragma: no cover
        torch_mod.backends.cudnn.enabled = False
    return torch_mod.device("cuda")


def _build_model(torch_mod, base: int = 16):
    """FC-Siam-diff-lite：共享编码器 + 逐层 |f1-f2| 差分融合解码器。"""
    nn = torch_mod.nn

    class _Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SiamDiffNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = base
            self.enc1 = _Block(2, b)
            self.enc2 = _Block(b, b * 2)
            self.enc3 = _Block(b * 2, b * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 4, 2, stride=2)
            self.dec2 = _Block(b * 4 + b * 2, b * 2)
            self.up1 = nn.ConvTranspose2d(b * 2, b * 2, 2, stride=2)
            self.dec1 = _Block(b * 2 + b, b)
            self.head = nn.Conv2d(b, 1, 1)

        def forward(self, x1, x2):
            e1a, e1b = self.enc1(x1), self.enc1(x2)
            e2a, e2b = self.enc2(self.pool(e1a)), self.enc2(self.pool(e1b))
            e3a, e3b = self.enc3(self.pool(e2a)), self.enc3(self.pool(e2b))
            d2 = torch_mod.cat([self.up2((e3a + e3b) / 2.0), torch_mod.abs(e2a - e2b)], 1)
            d2 = self.dec2(d2)
            d1 = torch_mod.cat([self.up1(d2), torch_mod.abs(e1a - e1b)], 1)
            d1 = self.dec1(d1)
            return torch_mod.sigmoid(self.head(d1))

    return SiamDiffNet()


def weights_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILENAME)


def _random_pair(rng: np.random.Generator, h: int = 64, w: int = 64):
    """生成一对合成双时相场景（0-3 个变化斑块），返回 (red1,nir1,red2,nir2,label)。"""
    gh, gw = 8, 8
    c1 = rng.choice(4, size=(gh, gw), p=[0.5, 0.25, 0.12, 0.13])
    c1 = np.repeat(np.repeat(c1, h // gh, axis=0), w // gw, axis=1)
    c2 = c1.copy()
    for _ in range(int(rng.integers(0, 4))):
        rh, rw = int(rng.integers(8, h // 2)), int(rng.integers(8, w // 2))
        y0, x0 = int(rng.integers(0, h - rh)), int(rng.integers(0, w - rw))
        c2[y0:y0 + rh, x0:x0 + rw] = int(rng.integers(0, 4))
    label = (c1 != c2).astype(np.float32)

    def to_refl(cls_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        red = np.empty((h, w), dtype=np.float32)
        nir = np.empty((h, w), dtype=np.float32)
        for k, name in enumerate(_CLASSES):
            rr, nn_ = _SPECTRA[name]
            m = cls_map == k
            red[m] = rr
            nir[m] = nn_
        red = red + rng.normal(0, 0.012, (h, w)).astype(np.float32)
        nir = nir + rng.normal(0, 0.020, (h, w)).astype(np.float32)
        return np.clip(red, 0, 1), np.clip(nir, 0, 1)

    red1, nir1 = to_refl(c1)
    red2, nir2 = to_refl(c2)
    # 时相间轻微辐射抖动（伪变化），网络需学会忽略
    eps = float(rng.normal(0, 0.006))
    red2 = np.clip(red2 + eps, 0, 1)
    nir2 = np.clip(nir2 + eps * 1.3, 0, 1)
    return red1, nir1, red2, nir2, label


def _dice_loss(torch_mod, prob, y, eps: float = 1e-6):
    p = prob.flatten(1)
    t = y.flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2.0 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def train_model(model, device, torch_mod, epochs: int = 40, n_pairs: int = 640,
                batch: int = 32, lr: float = 1.2e-3, seed: int = 42) -> Dict[str, Any]:
    """在合成双时相变化对上训练（GPU），返回训练元信息。"""
    torch_mod.manual_seed(seed)
    model = model.to(device)
    rng = np.random.default_rng(seed)
    pairs = [_random_pair(rng) for _ in range(n_pairs)]
    rng_h = np.random.default_rng(seed + 1)
    holdout = [_random_pair(rng_h) for _ in range(64)]

    opt = torch_mod.optim.Adam(model.parameters(), lr=lr)
    bce = torch_mod.nn.BCELoss()
    model.train()
    step = 0
    for _ in range(epochs):
        idx = rng.permutation(len(pairs))
        for s in range(0, len(idx), batch):
            chunk = [pairs[i] for i in idx[s:s + batch]]
            r1 = torch_mod.from_numpy(np.stack([p[0] for p in chunk]))[:, None]
            n1 = torch_mod.from_numpy(np.stack([p[1] for p in chunk]))[:, None]
            r2 = torch_mod.from_numpy(np.stack([p[2] for p in chunk]))[:, None]
            n2 = torch_mod.from_numpy(np.stack([p[3] for p in chunk]))[:, None]
            y = torch_mod.from_numpy(np.stack([p[4] for p in chunk]))[:, None]
            # 随机翻转增强
            if rng.random() < 0.5:
                r1, n1, r2, n2, y = (t.flip(3) for t in (r1, n1, r2, n2, y))
            r1, n1, r2, n2, y = (t.to(device) for t in (r1, n1, r2, n2, y))
            x1 = torch_mod.cat([r1, n1], 1).float()
            x2 = torch_mod.cat([r2, n2], 1).float()
            prob = model(x1, x2)
            loss = bce(prob, y) + _dice_loss(torch_mod, prob, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

    # holdout 评估（阈值 0.5）
    model.eval()
    recall_sum = fa_sum = 0.0
    with torch_mod.no_grad():
        for r1, n1, r2, n2, y in holdout:
            x1 = torch_mod.from_numpy(np.stack([r1, n1]))[None].to(device).float()
            x2 = torch_mod.from_numpy(np.stack([r2, n2]))[None].to(device).float()
            p = (model(x1, x2)[0, 0].cpu().numpy() >= 0.5)
            t = y.astype(bool)
            recall_sum += float(p[t].mean()) if t.any() else 1.0
            fa_sum += float(p[~t].mean()) if (~t).any() else 0.0
    meta = {
        "arch": "siam-diff-lite",
        "base": 16,
        "epochs": epochs,
        "n_pairs": n_pairs,
        "steps": step,
        "holdout_recall": round(recall_sum / len(holdout), 4),
        "holdout_false_alarm": round(fa_sum / len(holdout), 4),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "trained_at": _utc_now(),
        "framework": f"torch {torch_mod.__version__} / {torch_mod.cuda.get_device_name(0)}",
    }
    return meta


def _ensure_model() -> Tuple[Any, Any, Dict[str, Any]]:
    """加载随附权重；缺失时在 GPU 上训练并落盘。返回 (model, device, meta)。"""
    if _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"], _MODEL_CACHE["meta"]
    torch_mod = _require_torch()
    device = _cuda_device(torch_mod)
    model = _build_model(torch_mod)
    path = weights_path()
    if os.path.exists(path):
        ck = torch_mod.load(path, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        meta = dict(ck.get("meta", {}))
        meta["weights"] = os.path.basename(path)
    else:
        meta = train_model(model, device, torch_mod)
        meta["weights"] = "trained-on-the-fly (weights file missing)"
        torch_mod.save({"state_dict": model.state_dict(), "meta": meta}, path)
    model = model.to(device).eval()
    _MODEL_CACHE.update(model=model, device=device, meta=meta)
    return model, device, meta


def _normalize_bands(*bands: np.ndarray) -> Tuple[np.ndarray, ...]:
    """把反射率归一到 [0,1]：若数值明显是 0-10000 尺度则除以 10000。"""
    mx = max(float(np.nanmax(b)) for b in bands if b.size)
    factor = 10000.0 if mx > 1.5 else 1.0
    return tuple(np.clip(b.astype(np.float32) / factor, 0.0, 1.0) for b in bands)


def predict_change_prob(model, device, torch_mod,
                        red1: np.ndarray, nir1: np.ndarray,
                        red2: np.ndarray, nir2: np.ndarray) -> np.ndarray:
    """前向推理得到 [0,1] 变化概率（自动 pad 到 4 的倍数再裁回）。"""
    h, w = red1.shape
    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
    red1, nir1, red2, nir2 = _normalize_bands(red1, nir1, red2, nir2)

    def prep(red, nir):
        s = np.stack([red, nir]).astype(np.float32)
        if ph or pw:
            s = np.pad(s, ((0, 0), (0, ph), (0, pw)), mode="edge")
        return torch_mod.from_numpy(s)[None].to(device)

    with torch_mod.no_grad():
        prob = model(prep(red1, nir1), prep(red2, nir2))[0, 0]
    return prob[:h, :w].cpu().numpy()


# ---------------------------------------------------------------------------
# 完整变化检测流程
# ---------------------------------------------------------------------------
def detect_changes(
    red1: np.ndarray, nir1: np.ndarray,
    red2: np.ndarray, nir2: np.ndarray,
    prob_thresh: float = 0.5,
    scale: float = 10.0,
    min_area: int = 1,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """完整变化检测流程（Siamese FCN，GPU）。返回 (prob_map, binary_map, regions, info)。

    ``scale`` 为经典基线参数，主推理路径的概率由网络学习得到，此参数保留
    仅为 CLI 兼容。
    """
    a1, b1, a2, b2 = (np.asarray(x) for x in (red1, nir1, red2, nir2))
    if a1.shape != b1.shape or a2.shape != b2.shape or a1.shape != a2.shape:
        raise ValidationError(
            "bi-temporal band shape mismatch",
            t1=[list(a1.shape), list(b1.shape)], t2=[list(a2.shape), list(b2.shape)])
    torch_mod = _require_torch()
    model, device, meta = _ensure_model()
    prob = predict_change_prob(model, device, torch_mod, a1, b1, a2, b2)
    if valid_mask is not None:
        prob = np.where(valid_mask, prob, 0.0)
    binary = binary_change(prob, prob_thresh)
    labels, regions = change_regions(binary, min_area=min_area)

    # QA 诊断统计（NDVI 差分仅用于可解释性，不参与检测）
    n1 = ndvi(a1, b1)
    n2 = ndvi(a2, b2)
    diff = change_difference(n1, n2)
    info = {
        "mean_ndvi_t1": float(np.mean(n1)),
        "mean_ndvi_t2": float(np.mean(n2)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "n_change_regions": int(len(regions)),
        "change_fraction": float(np.mean(binary)),
        "backend": f"torch siamese-fcn ({meta.get('arch', 'siam-diff-lite')})",
        "device": str(device),
        "weights": str(meta.get("weights", "n/a")),
        "model_holdout_recall": meta.get("holdout_recall"),
        "model_holdout_false_alarm": meta.get("holdout_false_alarm"),
    }
    return prob, binary, regions, info


def pixel_box_to_geo(box: List[float], bbox: List[float], img_w: int, img_h: int) -> List[float]:
    w, s, e, n = bbox
    x1, y1, x2, y2 = [float(v) for v in box]
    lon1 = w + (x1 / img_w) * (e - w)
    lon2 = w + (x2 / img_w) * (e - w)
    lat1 = n - (y1 / img_h) * (n - s)
    lat2 = n - (y2 / img_h) * (n - s)
    return [min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)]


def regions_to_geojson(regions, bbox, img_w, img_h):
    feats = []
    for idx, r in enumerate(regions):
        gminx, gminy, gmaxx, gmaxy = pixel_box_to_geo(r["bbox_px"], bbox, img_w, img_h)
        ring = [[gminx, gminy], [gmaxx, gminy], [gmaxx, gmaxy],
                [gminx, gmaxy], [gminx, gminy]]
        feats.append({
            "type": "Feature", "id": int(idx),
            "properties": {"region_id": r["region_id"], "area_px": r["area_px"]},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------------------
# 合成数据：t1 全植被，t2 在中心区域砍伐（NDVI 下降）
# ---------------------------------------------------------------------------
def generate_synthetic(
    bbox: List[float],
    width: int = 64,
    height: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """返回 (red1, nir1, red2, nir2, change_truth, info)。

    t1：全幅高植被（高 NIR 低 Red）。t2：中心方形区域退化为裸土。
    change_truth 标记真实变化区。
    """
    rng = np.random.default_rng(seed)
    # 植被光谱：red 低、nir 高
    red1 = np.full((height, width), 0.05) + rng.normal(0, 0.01, (height, width))
    nir1 = np.full((height, width), 0.50) + rng.normal(0, 0.01, (height, width))
    red2 = red1.copy()
    nir2 = nir1.copy()

    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    # t2 中心区域退化为裸土：red 升高、nir 降低
    red2[y0:y1, x0:x1] = 0.25
    nir2[y0:y1, x0:x1] = 0.28

    truth = np.zeros((height, width), dtype=bool)
    truth[y0:y1, x0:x1] = True

    red1 = np.clip(red1, 0, 1).astype(np.float32)
    nir1 = np.clip(nir1, 0, 1).astype(np.float32)
    red2 = np.clip(red2, 0, 1).astype(np.float32)
    nir2 = np.clip(nir2, 0, 1).astype(np.float32)

    info = {"bbox": bbox, "width": width, "height": height,
            "change_box_px": [x0, y0, x1, y1]}
    return red1, nir1, red2, nir2, truth, info


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------
def write_geotiff(path: str, array: np.ndarray, bbox: List[float], nodata: float = -9999.0) -> None:
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    nb, h, w = arr.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": nb,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
        "nodata": nodata, "compress": "deflate",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(nb):
            dst.write(arr[b].astype("float32"), b + 1)


def read_geotiff(path: str) -> Tuple[np.ndarray, List[float]]:
    """读取栅格，返回 (cube, bbox)。保持既有签名（测试依赖）。"""
    cube, bbox, _ = read_geotiff_masked(path)
    return cube, bbox


def read_geotiff_masked(path: str) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """读取栅格，返回 (cube, bbox, valid_mask)；nodata 像元 mask 为 False。"""
    import rasterio
    if not os.path.exists(path):
        raise UsageError(f"input raster not found: {path}", path=path)
    with rasterio.open(path) as src:
        cube = src.read().astype(np.float32)
        b = src.bounds
        bbox = [b.left, b.bottom, b.right, b.top]
        try:
            mask = src.read_masks(1) > 0
        except Exception:
            mask = np.ones(cube.shape[1:], dtype=bool)
        nd = src.nodata
    if nd is not None:
        for k in range(cube.shape[0]):
            mask &= ~np.isclose(cube[k], nd)
    # 替换无效值为 0（网络输入），由 mask 控制后续统计
    cube = np.where(mask[None], cube, 0.0).astype(np.float32)
    return cube, bbox, mask


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(output_dir, args, outputs, qa, started_at, exit_code, bbox):
    if OutputManifest is None:
        return None
    cmd = " ".join([SKILL_NAME] + sys.argv[1:])
    man = OutputManifest(
        skill=SKILL_NAME, skill_version=VERSION, command=cmd,
        started_at=started_at, finished_at=_utc_now(), exit_code=exit_code,
        inputs={
            "input": getattr(args, "input", None),
            "input2": getattr(args, "input2", None),
            "synthetic": bool(getattr(args, "synthetic", False)),
        },
        outputs=[OutputFile(**o) for o in outputs],
        qa=qa,
        software={"python": sys.version.split()[0], "skill": SKILL_NAME},
    )
    path = os.path.join(output_dir, "output-manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def validate_bbox(bbox: List[float]) -> List[float]:
    """校验 bbox [W, S, E, N]；不合法抛 ValidationError（exit 6）。"""
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"bbox values must be numeric: {bbox}") from exc
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValidationError(f"bbox longitude out of range [-180, 180]: W={w}, E={e}")
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValidationError(f"bbox latitude out of range [-90, 90]: S={s}, N={n}")
    if w > e:
        raise ValidationError(
            f"bbox W ({w}) > E ({e}); antimeridian-crossing bbox is not supported — "
            "split the request into two bboxes on either side of +/-180")
    if s > n:
        raise ValidationError(f"bbox S ({s}) > N ({n})")
    return [w, s, e, n]


def _load_pair(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              List[float], str, Optional[np.ndarray], np.ndarray]:
    """按契约加载双时相数据。

    返回 (red1, nir1, red2, nir2, bbox, source, truth, valid_mask)。
    """
    bbox = list(args.bbox) if args.bbox else None
    if args.input and not args.synthetic:
        cube, file_bbox, mask1 = read_geotiff_masked(args.input)
        bbox = validate_bbox(bbox if bbox is not None else file_bbox)
        if args.input2:
            cube2, _, mask2 = read_geotiff_masked(args.input2)
            if cube2.shape[1:] != cube.shape[1:]:
                raise ValidationError(
                    "epoch rasters have different grid shape",
                    t1=list(cube.shape), t2=list(cube2.shape))
            if cube2.shape[0] < 2 or cube.shape[0] < 2:
                raise ValidationError("each epoch raster needs at least 2 bands (red, nir)")
            red1, nir1 = cube[0], cube[1]
            red2, nir2 = cube2[0], cube2[1]
        else:
            if cube.shape[0] < 4:
                raise ValidationError(
                    "single-file mode needs 4 bands [red1,nir1,red2,nir2]; "
                    "or provide --input2 for the second epoch")
            red1, nir1, red2, nir2 = cube[0], cube[1], cube[2], cube[3]
            mask2 = mask1
        valid = mask1 & mask2
        if not bool(valid.any()):
            raise ValidationError(
                "input raster(s) contain no valid (non-NoData) pixels")
        return red1, nir1, red2, nir2, bbox, args.input, None, valid
    # synthetic
    if bbox is None:
        raise UsageError("provide --bbox (synthetic mode) or --input <raster>")
    bbox = validate_bbox(bbox)
    red1, nir1, red2, nir2, truth, _ = generate_synthetic(bbox)
    valid = np.ones(red1.shape, dtype=bool)
    return red1, nir1, red2, nir2, bbox, "synthetic", truth, valid


def process(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    red1, nir1, red2, nir2, bbox, source_note, truth, valid = _load_pair(args)

    prob, binary, regions, info = detect_changes(
        red1, nir1, red2, nir2,
        prob_thresh=args.prob_thresh, scale=args.scale, min_area=args.min_area,
        valid_mask=valid,
    )
    h, w = prob.shape
    geojson = regions_to_geojson(regions, bbox, w, h)

    prob_path = os.path.join(output_dir, "change_probability.tif")
    write_geotiff(prob_path, prob.astype(np.float32), bbox)
    bin_path = os.path.join(output_dir, "change_binary.tif")
    write_geotiff(bin_path, binary.astype(np.float32), bbox)
    reg_path = os.path.join(output_dir, "change_regions.geojson")
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    qa: Dict[str, Any] = {
        "source": source_note,
        "backend": info["backend"],
        "device": info["device"],
        "weights": info["weights"],
        "model_holdout_recall": info["model_holdout_recall"],
        "model_holdout_false_alarm": info["model_holdout_false_alarm"],
        "n_change_regions": info["n_change_regions"],
        "change_fraction": info["change_fraction"],
        "mean_abs_diff": info["mean_abs_diff"],
    }
    if truth is not None:
        # 检出质量：变化区被正确标记的像元比例（召回）与虚警率
        qa["synthetic_recall"] = float(np.mean(binary[truth]))
        qa["synthetic_false_alarm"] = float(np.mean(binary[~truth]))

    outputs = [
        {"path": prob_path, "kind": "raster", "crs_epsg": 4326, "bbox_wgs84": bbox, "band_count": 1},
        {"path": bin_path, "kind": "raster", "crs_epsg": 4326, "bbox_wgs84": bbox, "band_count": 1},
        {"path": reg_path, "kind": "vector", "crs_epsg": 4326, "bbox_wgs84": bbox},
    ]
    man_path = write_manifest(output_dir, args, outputs, qa, started_at, 0, bbox)

    if not args.quiet:
        print(f"[{SKILL_NAME}] source: {source_note}")
        print(f"[{SKILL_NAME}] backend: {info['backend']} on {info['device']} "
              f"(weights: {info['weights']})")
        print(f"[{SKILL_NAME}] change regions: {info['n_change_regions']}  "
              f"change fraction: {info['change_fraction']:.4f}")
        if "synthetic_recall" in qa:
            print(f"[{SKILL_NAME}] synthetic recall: {qa['synthetic_recall']:.3f}  "
                  f"false alarm: {qa['synthetic_false_alarm']:.4f}")
        print(f"[{SKILL_NAME}] outputs: {prob_path}, {bin_path}, {reg_path}")
        if man_path:
            print(f"[{SKILL_NAME}] manifest: {man_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SKILL_NAME,
        description="Bi-temporal change detection with a Siamese FCN (FC-Siam-diff style, torch/CUDA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="geographic extent [minLon minLat maxLon maxLat]")
    p.add_argument("--input", help="epoch-1 GeoTIFF (red,nir) or 4-band [red1,nir1,red2,nir2]")
    p.add_argument("--input2", help="epoch-2 GeoTIFF (red,nir); optional with --input")
    p.add_argument("--prob-thresh", type=float, default=0.5, help="change probability threshold")
    p.add_argument("--scale", type=float, default=10.0,
                   help="legacy steepness parameter of the classical baseline; "
                        "kept for CLI compatibility, the learned model ignores it")
    p.add_argument("--min-area", type=int, default=1, help="drop change regions smaller than this (px)")
    p.add_argument("--synthetic", action="store_true", help="generate a synthetic pair (offline)")
    p.add_argument("--output-dir", default="./output", help="output directory")
    p.add_argument("--quiet", action="store_true", help="suppress console output")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return process(args)
    except GeoSkillError as exc:
        print(f"[{SKILL_NAME}] ERROR [{exc.kind}] {exc.message}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[{SKILL_NAME}] ERROR {exc}", file=sys.stderr)
        return to_exit_code(exc)


if __name__ == "__main__":
    sys.exit(main())
