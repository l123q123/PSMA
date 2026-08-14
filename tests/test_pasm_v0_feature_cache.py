import tempfile
import unittest
from pathlib import Path

import numpy as np

from pasm_v0.feature_cache import FeatureCache, cache_key, pair_cache_key


class FeatureCacheTests(unittest.TestCase):
    def test_round_trip_preserves_frozen_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = FeatureCache(directory)
            key = cache_key(Path("/data/a.jpg.jpg"))
            value = {
                "dense_logits": np.zeros((8, 8), dtype=np.float32),
                "feature_map": np.ones((8, 8, 32), dtype=np.float32),
                "tokens": np.full((16, 128), 2.0, dtype=np.float32),
                "nms_points": np.zeros((4, 2), dtype=np.float32),
                "nms_scores": np.ones((4,), dtype=np.float32),
            }
            cache.save(key, value)
            loaded = cache.load(key)
            self.assertEqual(set(loaded), set(value))
            for name in value:
                np.testing.assert_array_equal(loaded[name], value[name])

    def test_missing_entry_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                FeatureCache(directory).load("missing")

    def test_pair_cache_key_is_stable_and_distinct(self) -> None:
        self.assertEqual(pair_cache_key("pair-a"), pair_cache_key("pair-a"))
        self.assertNotEqual(pair_cache_key("pair-a"), pair_cache_key("pair-b"))
