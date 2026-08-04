"""Core algorithm tests for change-detection-dl."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conftest import mod


class TestNDVI:
    def test_pure_vegetation_high(self):
        red = np.full((4, 4), 0.05)
        nir = np.full((4, 4), 0.50)
        v = mod.ndvi(red, nir)
        expected = (0.50 - 0.05) / (0.50 + 0.05)
        np.testing.assert_allclose(v, expected, atol=1e-9)

    def test_bare_soil_low(self):
        v = mod.ndvi(np.full((4, 4), 0.25), np.full((4, 4), 0.28))
        assert np.all(v < 0.1)

    def test_zero_denominator_safe(self):
        v = mod.ndvi(np.zeros((4, 4)), np.zeros((4, 4)))
        np.testing.assert_array_equal(v, np.zeros((4, 4)))

    def test_shape_mismatch_raises(self):
        with pytest.raises(mod.ValidationError):
            mod.ndvi(np.zeros((4, 4)), np.zeros((3, 3)))

    def test_range_clipped(self):
        v = mod.ndvi(np.zeros((4, 4)), np.full((4, 4), 1.0))
        assert v.max() <= 1.0 and v.min() >= -1.0


class TestChangeDifference:
    def test_sign(self):
        t1 = np.full((4, 4), 0.8)
        t2 = np.full((4, 4), 0.2)  # 植被减少
        d = mod.change_difference(t1, t2)
        np.testing.assert_allclose(d, -0.6, atol=1e-9)

    def test_mismatch_raises(self):
        with pytest.raises(mod.ValidationError):
            mod.change_difference(np.zeros((4, 4)), np.zeros((5, 5)))


class TestChangeProbability:
    def test_zero_diff_zero_prob(self):
        p = mod.change_probability(np.zeros((4, 4)))
        np.testing.assert_allclose(p, 0.0, atol=1e-9)

    def test_monotonic_in_abs_diff(self):
        p_small = mod.change_probability(np.full((4, 4), 0.1))[0, 0]
        p_large = mod.change_probability(np.full((4, 4), 0.7))[0, 0]
        assert p_large > p_small

    def test_symmetric_sign(self):
        p_pos = mod.change_probability(np.full((4, 4), 0.5))[0, 0]
        p_neg = mod.change_probability(np.full((4, 4), -0.5))[0, 0]
        assert p_pos == pytest.approx(p_neg)

    def test_range_01(self):
        p = mod.change_probability(np.linspace(-1, 1, 100))
        assert p.min() >= 0.0 and p.max() <= 1.0


class TestBinaryAndRegions:
    def test_binary_threshold(self):
        prob = np.array([[0.1, 0.6], [0.9, 0.2]])
        b = mod.binary_change(prob, prob_thresh=0.5)
        np.testing.assert_array_equal(b, [[False, True], [True, False]])

    def test_two_regions(self):
        binary = np.zeros((20, 20), dtype=bool)
        binary[2:5, 2:5] = True
        binary[12:18, 12:18] = True
        labels, regions = mod.change_regions(binary)
        assert len(regions) == 2
        assert regions[0]["area_px"] == 36  # 6x6 排第一（降序）

    def test_min_area_filter(self):
        binary = np.zeros((20, 20), dtype=bool)
        binary[0, 0] = True            # 面积 1
        binary[10:15, 10:15] = True    # 面积 25
        _, regions = mod.change_regions(binary, min_area=5)
        assert len(regions) == 1


class TestDetectChanges:
    def test_injected_change_detected(self):
        """注入的变化区必须被检出（高召回、低虚警）。"""
        red1, nir1, red2, nir2, truth, _ = mod.generate_synthetic(
            [116, 39, 117, 40], seed=11
        )
        prob, binary, regions, info = mod.detect_changes(
            red1, nir1, red2, nir2, prob_thresh=0.5
        )
        recall = np.mean(binary[truth])
        false_alarm = np.mean(binary[~truth])
        assert recall > 0.9
        assert false_alarm < 0.05
        assert info["n_change_regions"] >= 1

    def test_no_change_empty(self):
        """两时相完全相同 -> 不应有变化。"""
        red = np.full((32, 32), 0.05)
        nir = np.full((32, 32), 0.5)
        prob, binary, regions, info = mod.detect_changes(red, nir, red, nir)
        assert info["n_change_regions"] == 0
        assert info["change_fraction"] == 0.0


class TestGeoJSON:
    def test_regions_to_geojson(self):
        binary = np.zeros((50, 50), dtype=bool)
        binary[10:20, 10:20] = True
        _, regions = mod.change_regions(binary)
        gj = mod.regions_to_geojson(regions, [116, 39, 117, 40], 50, 50)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 1
        ring = gj["features"][0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]


class TestGeoTiffIO:
    def test_roundtrip(self, tmp_path):
        arr = np.random.uniform(0, 1, (16, 16)).astype(np.float32)
        bbox = [116.0, 39.0, 117.0, 40.0]
        path = str(tmp_path / "chg.tif")
        mod.write_geotiff(path, arr, bbox)
        back, rb = mod.read_geotiff(path)
        np.testing.assert_allclose(back[0], arr, atol=1e-5)
        np.testing.assert_allclose(rb, bbox, atol=1e-6)

    def test_read_missing_raises(self):
        with pytest.raises(mod.UsageError):
            mod.read_geotiff("/nonexistent/x.tif")
