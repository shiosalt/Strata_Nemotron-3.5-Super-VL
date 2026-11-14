"""strata_control/nemotron35.py's sizes (UD-IQ3_XXS, UD-Q3_K_XL, UD-Q4_K_XL): each has three pinned shards with their size and SHA-256, a
folder of its own, a run config that names its first shard; UD-Q4_K_XL's config turns on the K-quant prompt kernels (STRATA_MMQ_KQ=1),
and every config waits up to an hour for the first start.  No GPU, no model files.

    python -m pytest tools/test_nemotron35_sizes.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strata_control import nemotron35 as N  # noqa: E402


class Sizes(unittest.TestCase):
    def test_three_sizes_with_pinned_shards(self):
        self.assertEqual(list(N.MODELS), ["UD-IQ3_XXS", "UD-Q3_K_XL", "UD-Q4_K_XL"])
        for m, d in N.MODELS.items():
            self.assertEqual(len(d["shards"]), 3)
            for s in d["shards"]:
                self.assertIn(m, s)
                size, sha = N.FILES[s]
                self.assertGreater(size, 1_000_000)
                self.assertEqual(len(sha), 64)
            self.assertEqual(d["subdir"], m)
            self.assertEqual(N.model_of_first(Path("x") / d["shards"][0]), m)
        self.assertEqual(round(N.size_row("UD-Q4_K_XL")["download_gb"], 1), 86.9)

    def test_find_set_tells_the_sizes_apart(self):
        with tempfile.TemporaryDirectory() as t:
            repo = Path(t) / N.REPO_FOLDER
            for m in ("UD-Q3_K_XL", "UD-Q4_K_XL"):
                f = repo / m
                f.mkdir(parents=True)
                (f / N.MODELS[m]["shards"][0]).write_bytes(b"GGUF")
                self.assertEqual(N.find_set(f), (m, f))
                self.assertEqual(N.find_set(f / N.MODELS[m]["shards"][0]), (m, f))
            self.assertEqual(N.folder_for(Path(t), "UD-Q4_K_XL"), repo / "UD-Q4_K_XL")

    def test_the_run_config_of_each_size(self):
        with tempfile.TemporaryDirectory() as t:
            for m in N.MODELS:
                first = Path(t) / N.MODELS[m]["shards"][0]
                cfg = N.build_config(first=first, exe=Path(t) / "strata.exe", tokenizer=Path(t) / "tok", out=Path(t), vision="off", lib_dirs=[])
                self.assertEqual(cfg["args"][1], str(first))
                self.assertEqual(cfg["engine_ready_s"], 3600)
                self.assertEqual(cfg.get("env"), {"STRATA_MMQ_KQ": "1"} if m == "UD-Q4_K_XL" else None)

    def test_the_ram_warning_follows_the_size(self):
        cards = [{"vram_gb": 16}, {"vram_gb": 16}]
        self.assertEqual(N.fit_lines(95.9, cards, "UD-Q4_K_XL"), [])
        self.assertTrue(any("UD-Q4_K_XL" in w for w in N.fit_lines(64, cards, "UD-Q4_K_XL")))
        self.assertEqual(N.fit_lines(64, cards, "UD-Q3_K_XL"), [])


if __name__ == "__main__":
    unittest.main()
