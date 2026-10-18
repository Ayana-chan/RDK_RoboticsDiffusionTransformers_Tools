#!/usr/bin/env python3
"""隔离检查实际导出源码的校准输入选择，不导入 Torch、HDF5 或编译工具链。"""

import ast
import logging
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import numpy as np


class H5Substitute:
    def __enter__(self):
        qpos = np.arange(12 * 14, dtype=np.float32).reshape(12, 14)
        return {"observations": {"qpos": qpos, "images": {}}, "action": qpos + 1000}

    def __exit__(self, *args):
        return False


class TorchSubstitute:
    def manual_seed(self, seed):
        self.seed = seed

    def load(self, path):
        assert Path(path).is_file()
        return int(Path(path).stem.rsplit("_", 1)[1])


def main():
    if not __debug__:
        raise RuntimeError("本测试使用 assert，不能用 python -O 运行")
    source = Path(__file__).resolve().parents[1] / "export_all.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    sampler = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_training_samples")
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    seeds = [n for n in entry.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) in {"np.random.seed", "torch.manual_seed"}]
    assert len(seeds) == 2
    logger = logging.getLogger("export-input-test")
    logger.disabled = True
    torch = TorchSubstitute()
    namespace = {"np": np, "os": os, "re": re, "torch": torch, "logger": logger,
                 "h5py": SimpleNamespace(File=lambda *a: H5Substitute()),
                 "opt": SimpleNamespace(seed=20260913)}
    exec(compile(ast.Module(body=[sampler], type_ignores=[]), str(source), "exec"), namespace)
    seed_code = compile(ast.Module(body=seeds, type_ignores=[]), str(source), "exec")
    with tempfile.TemporaryDirectory(prefix="rdt-export-input-") as directory:
        root = Path(directory)
        for index in range(8):
            episode = root / f"episode_{index}"
            instructions = episode / "instructions"
            instructions.mkdir(parents=True)
            (episode / "episode.hdf5").write_bytes(b"hdf5-substitute")
            for instruction in range(3):
                (instructions / f"lang_embed_{instruction}.pt").write_text("torch-substitute")
                (instructions / f"txt_lang_embed_{instruction}.txt").write_text(str(instruction))
        signatures = []
        for _ in range(2):
            exec(seed_code, namespace)
            assert torch.seed == 20260913
            samples = namespace["get_training_samples"](str(root), num_samples=8)
            assert len(samples) == 8
            assert any(sample["step"] >= 3 for sample in samples), "未覆盖时间步超出指令数量的情形"
            signature = []
            for sample in samples:
                step = sample["step"]
                expected = np.arange(12 * 14, dtype=np.float32).reshape(12, 14)[step]
                assert np.array_equal(sample["joints"], expected), "校准输入使用了动作标签而非同帧 qpos"
                assert sample["lang_str"] == str(sample["lang_embed"]), "指令文本和 embedding 不匹配"
                signature.append((sample["source"], step, sample["lang_embed"], sample["joints"].tobytes()))
            signatures.append(signature)
        assert signatures[0] == signatures[1], "固定 seed 未复现抽样结果"
    print("通过：直接源码的随机种子、独立指令索引和同帧 qpos 选择；未执行真实导出或 HBM 编译。")


if __name__ == "__main__":
    main()
