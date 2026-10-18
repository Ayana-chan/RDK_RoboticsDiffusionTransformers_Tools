#!/usr/bin/env python3
"""直接核验当前运行时源码的 step 契约；仅使用 NumPy 替身，不执行模型或 BPU。"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np



class Tensor(np.ndarray):
    """只实现原 step 用到的张量接口，数值内核由 NumPy 替代。"""

    def __new__(cls, value):
        return np.asarray(value).view(cls)

    @property
    def device(self):
        return "cpu"

    def float(self):
        return Tensor(self.astype(np.float32, copy=False))

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self))

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self, dim))

    def expand(self, *shape):
        shape = tuple(current if requested == -1 else requested for current, requested in zip(self.shape, shape))
        return Tensor(np.broadcast_to(self, shape))

    def to(self, dtype):
        return Tensor(self.astype(dtype, copy=False))


class TorchSubstitute:
    float32 = np.float32
    bool = np.bool_

    def __init__(self, events=None):
        self.noise = None
        self.noise_calls = 0
        self.events = events
        self.noise_error = None

    @staticmethod
    def from_numpy(value):
        return Tensor(value)

    @staticmethod
    def tensor(value):
        return Tensor(value)

    @staticmethod
    def zeros(shape, dtype, device=None):
        return Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype):
        return Tensor(np.ones(shape, dtype=dtype))

    @staticmethod
    def cat(values, dim=0):
        return Tensor(np.concatenate(values, axis=dim))

    def randn(self, size, dtype):
        assert tuple(size) == self.noise.shape
        self.noise_calls += 1
        if self.events is not None:
            self.events.append(("noise",))
        if self.noise_error is not None:
            raise self.noise_error
        return Tensor(self.noise.copy().astype(dtype))


class AdaptorSubstitute:
    def __init__(self, name="adaptor", events=None):
        self.name, self.events = name, events
        self.inputs = []
        self.error = None

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, names, values):
        value = next(iter(values.values()))
        self.inputs.append(value.copy())
        if self.events is not None:
            self.events.append((self.name,))
        if self.error is not None:
            raise self.error
        output = value[..., :8] + value.mean(axis=-1, keepdims=True)
        # 故意提供非 C 连续输出，检查原有 contiguous 语义仍被保留。
        return [np.asfortranarray(output.astype(np.float32))]


class DiTSubstitute:
    """模拟同步消费输入，保存全部六个输入；没有设备推理或设备计时。"""

    def __init__(self, events=None):
        self.inputs = []
        self.pointers = []
        self.events = events

    def __call__(self, arrays):
        if self.events is not None:
            self.events.append(("dit",))
        assert len(arrays) == 6
        assert all(array.flags.c_contiguous for array in arrays)
        assert [array.dtype for array in arrays] == [np.float32, np.int32, np.int32, np.float32, np.float32, np.float32]
        self.inputs.append([array.copy() for array in arrays])
        self.pointers.append([array.ctypes.data for array in arrays])
        x, freq, timestep, language, image, mask = arrays
        base = x[:, -64:, :].mean(axis=-1, keepdims=True)
        # 六个输入都参与替身输出，避免测试只检查最终 shape。
        correction = np.float32(sum(float(array.mean()) for array in (freq, timestep, language, image, mask)) / 4096)
        return [np.repeat(base + correction, 128, axis=-1).astype(np.float32)]


class SchedulerSubstitute:
    def __init__(self, events=None):
        self.events = events

    def set_timesteps(self, count):
        assert count == 5
        if self.events is not None:
            self.events.append(("set_timesteps",))
        self.timesteps = [Tensor(np.int64(value)) for value in (999, 799, 599, 400, 200)]

    def step(self, output, timestep, noisy_action):
        if self.events is not None:
            self.events.append(("solver", int(timestep)))
        # 本替身仅检查循环依赖；它不是 DPMSolver 的数值模拟器。
        return SimpleNamespace(prev_sample=Tensor(noisy_action * np.float32(.75) + output * np.float32(.25)))


class CV2Substitute:
    """只用于追踪同一预处理表达式；近邻采样不冒充真实 OpenCV 插值。"""

    BORDER_CONSTANT = 0

    def __init__(self):
        self.resize_calls = 0
        self.error_at = None
        self.error = RuntimeError("注入预处理失败")

    def resize(self, image, shape):
        self.resize_calls += 1
        if self.resize_calls == self.error_at:
            raise self.error
        width, height = shape
        rows = np.arange(height) * image.shape[0] // height
        columns = np.arange(width) * image.shape[1] // width
        return image[rows[:, None], columns[None, :]]

    @staticmethod
    def copyMakeBorder(image, top, bottom, left, right, mode, value):
        assert (top, bottom, left, right, mode, value) == (48, 48, 0, 0, 0, (127, 127, 127))
        return np.pad(image, ((top, bottom), (left, right), (0, 0)), constant_values=127)


class SigLIPSubstitute:
    """同步复制 start 输入、显式维护在途槽；不运行任何设备模型。"""

    def __init__(self, events):
        self.events = events
        self.inputs = []
        self.pending = {}
        self.start_error_id = None
        self.wait_error_id = None
        self.error = RuntimeError("注入 SDK 失败")

    def start(self, arrays, task_id):
        assert task_id not in self.pending and 0 <= task_id < 6
        assert len(arrays) == 1 and arrays[0].shape == (1, 3, 384, 384)
        assert arrays[0].dtype == np.float32 and arrays[0].flags.c_contiguous
        value = arrays[0].copy()
        self.inputs.append((task_id, value))
        self.events.append(("start", task_id))
        self.pending[task_id] = value
        # 模拟 native 已创建资源但 start 抛错的未知状态，Python 不应重试此槽。
        if task_id == self.start_error_id:
            raise self.error

    def wait(self, task_id):
        assert task_id in self.pending
        self.events.append(("wait", task_id))
        if task_id == self.wait_error_id:
            raise self.error
        value = self.pending.pop(task_id)
        channels = value.mean(axis=(2, 3), dtype=np.float32)
        features = np.tile(channels, (2, 384)).reshape(1, 2, 1152)
        return [features]


def policy_from_source(source, *, vision_pipeline=False):
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BPU_RDT_Policy":
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}
            if names & {"STATE_VEC_IDX_MAPPING", "STATE_VEC_LEN", "AGILEX_STATE_INDICES"}:
                selected.append(node)
    events = []
    torch = TorchSubstitute(events)
    cv2 = CV2Substitute()
    namespace = {"np": np, "torch": torch, "cv2": cv2, "logger": logging.getLogger("runtime-contract-test"), "time": time.time}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "runtime-contract-substitute", "exec"), namespace)
    cls = namespace["BPU_RDT_Policy"]
    model = cls.__new__(cls)
    model.pred_horizon = 64
    model.action_dim = 128
    model.num_inference_timesteps = 5
    model.noise_scheduler_sample = SchedulerSubstitute(events)
    model.state_adaptor1 = AdaptorSubstitute("state1", events)
    model.state_adaptor2 = AdaptorSubstitute("state2", events)
    model.lang_adaptor = AdaptorSubstitute()
    image_adaptor = AdaptorSubstitute("img_adaptor", events)
    model.img_adaptor = lambda inputs: image_adaptor.run([], {"input": inputs[0]})
    model.vision_emb = model.vision_emb_single if vision_pipeline else lambda images: Tensor(np.stack(images))
    model.siglip = SigLIPSubstitute(events)
    model.dit = DiTSubstitute(events)
    model._test_events = events
    model._test_cv2 = cv2
    return model, torch


def check_step_equivalence(source, candidate, *, reuse_conditions=True):
    baseline, baseline_torch = policy_from_source(source)
    optimized, optimized_torch = policy_from_source(candidate)
    rng = np.random.default_rng(20261017)
    noise = rng.standard_normal((1, 64, 128)).astype(np.float32)
    images = [rng.standard_normal((2, 1152)).astype(np.float32) for _ in range(6)]
    joints = rng.standard_normal(14).astype(np.float32)
    checks = 0
    first_action = None
    # A→B→C→A 覆盖语言长度、图像条件、频率与关节更新，防止跨请求污染。
    for length, frequency, delta in ((1, 25, 0), (64, 20, 1), (4, 30, -1), (1, 25, 0)):
        language = np.arange(length * 4096, dtype=np.float32).reshape(1, length, 4096) / np.float32(8192)
        outputs = []
        for model, torch in ((baseline, baseline_torch), (optimized, optimized_torch)):
            torch.noise = noise
            torch.noise_calls = 0
            model.control_frequency = frequency
            model.set_lang_condition(language)
            model.dit.inputs.clear()
            model.dit.pointers.clear()
            output = model.step([image + np.float32(delta) for image in images], joints + np.float32(delta))
            assert output.shape == (64, 14) and np.isfinite(output).all()
            assert torch.noise_calls == 1 and len(model.dit.inputs) == 5
            outputs.append(output)
        for original_step, changed_step in zip(baseline.dit.inputs, optimized.dit.inputs):
            for original, changed in zip(original_step, changed_step):
                assert original.shape == changed.shape and original.dtype == changed.dtype
                assert original.tobytes() == changed.tobytes(), "DiT 输入发生数值变化"
                checks += 1
        assert outputs[0].tobytes() == outputs[1].tobytes(), "最终动作发生数值变化"
        if reuse_conditions:
            for index in (1, 3, 4, 5):
                assert len({pointers[index] for pointers in optimized.dit.pointers}) == 1, "条件数组未在单 step 内复用"
        if first_action is None:
            first_action = outputs[1].copy()
        elif delta == 0:
            assert first_action.tobytes() == outputs[1].tobytes(), "A→B→C→A 发生状态污染"
    return checks


def check_vision_pipeline(source, variants):
    """保留真实方法调用链，核对六图身份、五轮状态依赖和 CPU/等待先后。"""
    rng = np.random.default_rng(20261018)
    images = [rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8) for _ in range(6)]
    joints = rng.standard_normal(14).astype(np.float32)
    noise = rng.standard_normal((1, 64, 128)).astype(np.float32)
    snapshots = {}
    checks = {name: {"vision_inputs": 0, "dit_inputs": 0, "policy_steps": 0} for name in variants}
    for name, code in {"baseline": source, **variants}.items():
        model, torch = policy_from_source(code, vision_pipeline=True)
        snapshots[name] = []
        first_action = None
        for length, frequency, delta in ((1, 25, 0), (64, 20, 1), (4, 30, -1), (1, 25, 0)):
            model.control_frequency = frequency
            model.set_lang_condition(np.arange(length * 4096, dtype=np.float32).reshape(1, length, 4096) / np.float32(8192))
            torch.noise, torch.noise_calls = noise, 0
            model._test_events.clear()
            model.siglip.inputs.clear()
            model.state_adaptor1.inputs.clear()
            model.state_adaptor2.inputs.clear()
            model.dit.inputs.clear()
            model.dit.pointers.clear()
            current_images = [((image.astype(np.int16) + delta) % 256).astype(np.uint8) for image in images]
            output = model.step(current_images, joints + np.float32(delta))
            events = model._test_events
            starts = [i for i, event in enumerate(events) if event[0] == "start"]
            waits = [i for i, event in enumerate(events) if event[0] == "wait"]
            state2 = [i for i, event in enumerate(events) if event[0] == "state2"]
            dits = [i for i, event in enumerate(events) if event[0] == "dit"]
            solvers = [i for i, event in enumerate(events) if event[0] == "solver"]
            assert [events[i][1] for i in starts] == list(range(6))
            assert [events[i][1] for i in waits] == list(range(6))
            assert max(starts) < min(waits) and not model.siglip.pending
            assert torch.noise_calls == 1 and events.count(("set_timesteps",)) == 1
            assert events.count(("state1",)) == 1 and len(state2) == len(dits) == len(solvers) == 5
            assert max(waits) < events.index(("img_adaptor",)) < dits[0]
            if name in ("overlap", "combined"):
                assert max(starts) < events.index(("state1",)) < state2[0] < min(waits)
            else:
                assert max(waits) < events.index(("state1",)) < state2[0]
            for index in range(5):
                assert state2[index] < dits[index] < solvers[index]
                if index:
                    assert solvers[index - 1] < state2[index]
            if first_action is None:
                first_action = output.copy()
            elif delta == 0:
                assert output.tobytes() == first_action.tobytes()
            if name in ("candidate", "combined"):
                for index in (1, 3, 4, 5):
                    assert len({pointers[index] for pointers in model.dit.pointers}) == 1
            snapshot = ([(task_id, hashlib.sha256(value.tobytes()).hexdigest()) for task_id, value in model.siglip.inputs], copy.deepcopy(model.dit.inputs),
                        copy.deepcopy(model.state_adaptor2.inputs), output.copy())
            snapshots[name].append(snapshot)
            if name == "baseline":
                continue
            reference = snapshots["baseline"][len(snapshots[name]) - 1]
            for (expected_id, expected), (actual_id, actual) in zip(reference[0], snapshot[0]):
                assert actual_id == expected_id and actual == expected
                checks[name]["vision_inputs"] += 1
            for original_step, changed_step in zip(reference[1], snapshot[1]):
                for original, changed in zip(original_step, changed_step):
                    assert original.shape == changed.shape and original.dtype == changed.dtype
                    assert original.tobytes() == changed.tobytes(), f"{name} 的 DiT 输入变化"
                    checks[name]["dit_inputs"] += 1
            assert all(a.tobytes() == b.tobytes() for a, b in zip(reference[2], snapshot[2]))
            assert reference[3].tobytes() == output.tobytes()
            checks[name]["policy_steps"] += 1
    return checks


def check_overlap_failures(variants):
    checks = 0
    # 屏蔽预期故障的 traceback；仍检查原始异常对象与每个 task 的回收序列。
    logger = logging.getLogger("runtime-contract-test")
    old_disabled = logger.disabled
    logger.disabled = True
    try:
        for name, source in variants.items():
            for failure in ("preprocess", "state1", "noise", "state2", "submit", "collect", "drain"):
                model, torch = policy_from_source(source, vision_pipeline=True)
                model.control_frequency = 25
                model.set_lang_condition(np.zeros((1, 4, 4096), dtype=np.float32))
                torch.noise = np.zeros((1, 64, 128), dtype=np.float32)
                images = [np.full((3, 4, 3), i, dtype=np.uint8) for i in range(6)]
                expected = RuntimeError(f"注入 {failure} 失败")
                if failure == "preprocess":
                    model._test_cv2.error_at, model._test_cv2.error = 3, expected
                elif failure in ("state1", "state2"):
                    getattr(model, f"state_adaptor{failure[-1]}").error = expected
                elif failure == "noise":
                    torch.noise_error = expected
                elif failure == "submit":
                    model.siglip.start_error_id, model.siglip.error = 2, expected
                elif failure == "collect":
                    model.siglip.wait_error_id, model.siglip.error = 2, expected
                else:
                    model.state_adaptor1.error = expected
                    model.siglip.wait_error_id = 2
                try:
                    model.step(images, np.zeros(14, dtype=np.float32))
                except RuntimeError as error:
                    assert error is expected, f"{name}/{failure} 没有保留原异常"
                else:
                    raise AssertionError(f"{name}/{failure} 未失败")
                waited = [event[1] for event in model._test_events if event[0] == "wait"]
                assert waited == (list(range(2)) if failure in ("preprocess", "submit") else list(range(6)))
                assert not model.dit.inputs
                poisoned = failure in ("submit", "collect", "drain")
                assert bool(getattr(model, "_vision_sdk_failed", False)) == poisoned
                assert set(model.siglip.pending) == ({2} if poisoned else set())
                if poisoned:
                    before = model._test_events.copy()
                    try:
                        model.step(images, np.zeros(14, dtype=np.float32))
                    except RuntimeError as error:
                        assert "必须重建运行时" in str(error)
                    else:
                        raise AssertionError("SDK 状态未知的实例被再次使用")
                    assert model._test_events == before
                else:
                    # 纯 Python/CPU 失败已经排空 SDK 任务，修复输入后可以复用同一实例。
                    model._test_cv2.error_at = None
                    model.state_adaptor1.error = model.state_adaptor2.error = None
                    torch.noise_error = None
                    recovered = model.step(images, np.zeros(14, dtype=np.float32))
                    assert recovered.shape == (64, 14) and np.isfinite(recovered).all()
                    assert not model.siglip.pending
                checks += 1
            # 独立视觉方法仍能直接调用；其部分提交异常也必须 finally 回收。
            model, _ = policy_from_source(source, vision_pipeline=True)
            model._test_cv2.error_at = 3
            try:
                model.vision_emb_single(images)
            except RuntimeError as error:
                assert error is model._test_cv2.error
            else:
                raise AssertionError("独立视觉预处理未传播异常")
            assert not model.siglip.pending
            assert [event[1] for event in model._test_events if event[0] == "wait"] == [0, 1]
            checks += 1
            model._test_cv2.error_at = None
            model._test_events.clear()
            result = model.vision_emb_single(images)
            assert result.shape == (6, 2, 1152) and not model.siglip.pending
            assert model._test_events == [("start", i) for i in range(6)] + [("wait", i) for i in range(6)]
            checks += 1
    finally:
        logger.disabled = old_disabled
    return checks


def main():
    if not __debug__:
        raise RuntimeError("本测试使用 assert，不能用 python -O 运行")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("overlap", "combined", "candidate"), required=True,
                        help="明确指定被测方案，防止把条件复用或重排检查静默跳过")
    parser.add_argument("--runtime-source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "board/BPU_RDT_policy.py")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_commit = "e84abb09bb4d79110cebc376711e78a2a0d7567a"
    reference = subprocess.run(
        ["git", "-C", str(root), "show", baseline_commit + ":board/BPU_RDT_policy.py"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    baseline_sha = hashlib.sha256(reference).hexdigest()
    if baseline_sha != "820932730847ebec79cfb48ca5bd5d7d9beb416a6c98894f1764c4508a414bb0":
        raise ValueError("固定基线源码 SHA256 不匹配")
    source, candidate = reference.decode("utf-8"), args.runtime_source.read_text(encoding="utf-8")
    compile(candidate, str(args.runtime_source), "exec")
    # 替身不执行构造函数和模块导入；静态检查确保这些部分没有随本次优化改变。
    before, after = ast.parse(source), ast.parse(candidate)
    allowed = {"step", "vision_emb_single", "vision_emb_submit", "vision_emb_collect", "vision_emb_drain"}
    for tree in (before, after):
        policy = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BPU_RDT_Policy")
        policy.body = [node for node in policy.body if not isinstance(node, ast.FunctionDef) or node.name not in allowed]
    assert ast.dump(before) == ast.dump(after), "step/单板视觉之外出现未覆盖改动"
    checks = check_step_equivalence(source, candidate, reuse_conditions=args.variant != "overlap")
    pipeline = check_vision_pipeline(source, {args.variant: candidate})
    failures = check_overlap_failures({args.variant: candidate}) if args.variant != "candidate" else 0
    print(json.dumps({
        "passed": True,
        "scope": "直接检查当前源码的 NumPy/OpenCV/SDK 替身契约；不是真实模型数值或性能验收",
        "variant": args.variant,
        "baseline_commit": baseline_commit,
        "baseline_sha256": baseline_sha,
        "runtime_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "synchronous_callback_input_checks": checks,
        "vision_pipeline_checks": pipeline,
        "overlap_failure_checks": failures,
        "board_validation": "本测试未执行",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
