"""基于torch.profiler的模型性能分析工具

用法示例:
    profiler = ModelProfiler(model, example_inputs)
    profiler.run()          # torch.profiler采集，按周期输出算子级热点表
    profiler.benchmark()    # wall-clock端到端计时（无profiler开销，交叉验证用）

    onnx_prof = OnnxProfiler("model.onnx")
    onnx_prof.run()         # onnxruntime自带profiler，输出按算子类型聚合的热点表
    onnx_prof.benchmark()   # 与ModelProfiler.benchmark()同口径的wall-clock计时
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule, tensorboard_trace_handler


class ModelProfiler:
    """设备自适应的模型性能分析器

    设备相关的差异点集中在auto_device/sync/activities/sort_by四处，
    接入新后端（如NPU/XPU等）时只需扩展这几个方法。

    注意：ProfilerActivity暂不支持MPS，MPS上只能采集CPU侧分发耗时，
    GPU kernel级分析需配合torch.mps.profiler + Metal System Trace。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: torch.Tensor,
        device: str = None,
        # schedule参数：skip_first/warmup阶段代替手动预热，不计入结果
        skip_first: int = 10,
        wait: int = 5,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 2,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        row_limit: int = 10,
        trace_dir: str = None,  # 指定后将trace保存为tensorboard格式
    ):
        self.device = device or self.auto_device()
        self.model = model.to(self.device).eval()
        self.inputs = example_inputs.to(self.device)

        self.schedule = schedule(
            skip_first=skip_first, 
            wait=wait, 
            warmup=warmup, 
            active=active, 
            repeat=repeat
        )
        # schedule是由prof.step()驱动的状态机，必须走满步数才能完成全部采集周期
        self.num_steps = skip_first + (wait + warmup + active) * repeat

        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.row_limit = row_limit
        self.trace_dir = trace_dir

        self.key_averages = []  # 每个采集周期的key_averages结果

    
    @staticmethod
    def auto_device() -> str:
        """
        自动选择可用设备：CUDA > MPS > CPU
        设备相关的差异点（新后端在此扩展）
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def sync(self):
        # MPS/CUDA执行是异步的，同步后CPU耗时才能反映真实端到端时间
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    @property
    def activities(self) -> list:
        # ProfilerActivity暂不支持MPS，只有CUDA能采集GPU kernel时间线
        acts = [ProfilerActivity.CPU]
        if self.device == "cuda":
            acts.append(ProfilerActivity.CUDA)
        return acts

    @property
    def sort_by(self) -> str:
        return "self_cuda_time_total" if self.device == "cuda" else "self_cpu_time_total"

    @torch.no_grad()
    def _forward(self):
        self.model(self.inputs)

    def _on_trace_ready(self, p: torch.profiler.profile):
        # 每完成一个active采集周期被回调一次（repeat几次就回调几次）；
        # repeat>1时profiler只保留最后一个周期，因此在回调里逐周期落数据
        averages = p.key_averages()
        self.key_averages.append(averages)
        print(f"===== trace ready at step {p.step_num} =====")
        print(averages.table(sort_by=self.sort_by, row_limit=self.row_limit))
        if self.trace_dir:
            tensorboard_trace_handler(self.trace_dir)(p)

    def run(self) -> list:
        """执行profiler采集，返回各周期的key_averages列表"""
        self.key_averages = []
        print(f"Profiling on device: {self.device}")

        with profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        ) as prof:
            for _ in range(self.num_steps):
                with record_function("model_inference"):
                    self._forward()
                    self.sync()
                prof.step()

        return self.key_averages

    def benchmark(self, warmup: int = 10, iters: int = 50) -> float:
        """wall-clock端到端计时（无profiler记录开销），返回单轮平均耗时(ms)

        profiler对每次算子调用都有记录开销，最终优化收益应以此为准。
        """
        for _ in range(warmup):
            self._forward()
        self.sync()

        start = time.perf_counter()
        for _ in range(iters):
            self._forward()
        self.sync()
        avg_ms = (time.perf_counter() - start) / iters * 1000

        print(f"[benchmark] device={self.device}, iters={iters}, avg={avg_ms:.3f}ms/iter")
        return avg_ms


class OnnxProfiler:
    """基于onnxruntime自带profiler的ONNX模型性能分析器

    torch.profiler挂在aten算子分发层，看不见ORT运行时的执行；
    这里用SessionOptions.enable_profiling采集逐节点耗时（chrome trace格式），
    解析后聚合出与key_averages().table()等价的热点表。
    """

    def __init__(
        self,
        onnx_path: str,
        providers: list = None,   # 默认CPU EP；Mac上可传['CoreMLExecutionProvider', 'CPUExecutionProvider']
        warmup: int = 10,         # 预热轮次（trace中按model_run时间戳过滤掉，不计入统计）
        active: int = 20,         # 计入统计的轮次
        row_limit: int = 10,
    ):
        import onnxruntime as ort  # 延迟导入，不强依赖
        self.ort = ort

        self.onnx_path = onnx_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.warmup = warmup
        self.active = active
        self.row_limit = row_limit

        # 用一个不开profiling的session探测输入元信息，并供benchmark复用
        self.session = ort.InferenceSession(onnx_path, providers=self.providers)
        self.input_feed = {
            i.name: np.random.randn(*i.shape).astype(np.float32)
            for i in self.session.get_inputs()
        }

    def _run_once(self, session):
        session.run(None, self.input_feed)  # ORT的run是同步返回的，无需手动sync

    def run(self) -> list:
        """采集逐节点耗时，打印按算子类型聚合的热点表，返回聚合结果列表"""
        so = self.ort.SessionOptions()
        so.enable_profiling = True
        # trace落在onnx模型旁边，避免散落在当前工作目录
        so.profile_file_prefix = os.path.splitext(self.onnx_path)[0] + "_ort_profile"
        sess = self.ort.InferenceSession(self.onnx_path, so, providers=self.providers)

        for _ in range(self.warmup + self.active):
            self._run_once(sess)
        trace_file = sess.end_profiling()

        with open(trace_file) as f:
            events = json.load(f)

        # 按model_run事件的时间戳切掉warmup轮次，只统计active窗口内的节点事件
        run_starts = sorted(e["ts"] for e in events if e.get("name") == "model_run")
        ts_threshold = run_starts[self.warmup] if len(run_starts) > self.warmup else 0

        # 节点级kernel事件: cat==Node且name以_kernel_time结尾，args里有op_name/provider
        stats = defaultdict(lambda: {"total_us": 0.0, "count": 0, "providers": set()})
        for e in events:
            if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
                continue
            if e["ts"] < ts_threshold:
                continue
            op = e["args"].get("op_name", "unknown")
            stats[op]["total_us"] += e["dur"]
            stats[op]["count"] += 1
            stats[op]["providers"].add(e["args"].get("provider", "?"))

        rows = [
            {
                "op_type": op,
                "total_ms": s["total_us"] / 1000,
                "avg_us": s["total_us"] / s["count"],
                "calls": s["count"],
                "calls_per_run": s["count"] // self.active,
                "providers": ",".join(sorted(s["providers"])),
            }
            for op, s in stats.items()
        ]
        rows.sort(key=lambda r: r["total_ms"], reverse=True)

        grand_total = sum(r["total_ms"] for r in rows)
        print(f"===== [OnnxProfiler] {self.onnx_path} providers={self.providers} "
              f"(active={self.active} runs) =====")
        header = f"{'Op Type':<24}{'Total':>12}{'Total %':>10}{'Avg/call':>12}{'# Calls':>10}{'Per Run':>10}  Provider"
        print(header)
        print("-" * len(header))
        for r in rows[: self.row_limit]:
            print(f"{r['op_type']:<24}{r['total_ms']:>10.3f}ms"
                  f"{r['total_ms'] / grand_total * 100:>9.1f}%"
                  f"{r['avg_us']:>10.1f}us{r['calls']:>10}{r['calls_per_run']:>10}  {r['providers']}")
        print("-" * len(header))
        print(f"节点耗时合计: {grand_total:.3f}ms, 单轮平均: {grand_total / self.active:.3f}ms "
              f"(trace: {trace_file})")
        return rows

    def benchmark(self, warmup: int = 10, iters: int = 50) -> float:
        """wall-clock端到端计时（无profiling开销），返回单轮平均耗时(ms)"""
        for _ in range(warmup):
            self._run_once(self.session)

        start = time.perf_counter()
        for _ in range(iters):
            self._run_once(self.session)
        avg_ms = (time.perf_counter() - start) / iters * 1000

        print(f"[benchmark] providers={self.providers}, iters={iters}, avg={avg_ms:.3f}ms/iter")
        return avg_ms



def model_fuse(model:torch.nn.Module, device, input_size=(1, 3, 224, 224)):
    from torch.fx.experimental.optimization import fuse
    
    # fx融合必须在eval模式下进行，否则BN统计量还会更新，折叠不等价
    baseline = model.eval().to(device)
    fused = fuse(baseline)
     
    # 正确性校验：融合前后输出应数值一致
    example_inputs = torch.randn(input_size, device=device)
    with torch.no_grad():
        out_a, out_b = baseline(example_inputs), fused(example_inputs)
    # max_diff = (out_a - out_b).abs().max().item()
    # print(f"融合前后输出最大误差: {max_diff:.2e}")
    # assert max_diff < 1e-3, "融合后输出不一致！"
    if torch.allclose(out_a, out_b, atol=1e-3):
        raise ValueError("融合后输出不一致！")

    # 确认BN确实被消除
    n_bn_base = sum(isinstance(m, torch.nn.BatchNorm2d) for m in baseline.modules())
    n_bn_fused = sum(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
    print(f"BatchNorm2d层数: baseline={n_bn_base} -> fused={n_bn_fused}\n")
    return fused



if __name__ == "__main__":
    import os

    import torchvision.models as models

    model = models.resnet18()
    example_inputs = torch.randn(1, 3, 224, 224)

    onnx_path = "checkpoints/resnet18.onnx"
    if not os.path.exists(onnx_path):
        torch.onnx.export(
            model.cpu().eval(), example_inputs, onnx_path,
            input_names=["input"], output_names=["output"], opset_version=17,
        )
        print(f"exported: {onnx_path}")

    # ---- torch路径 ----
    profiler = ModelProfiler(model=model, 
                             example_inputs=example_inputs)
    profiler.run()
    torch_ms = profiler.benchmark()

    # ---- ONNX路径：同一模型同一输入配置导出后对比 ----
    providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
    onnx_prof = OnnxProfiler(onnx_path, providers=providers)
    onnx_prof.run()
    ort_ms = onnx_prof.benchmark()

    print("\n===== 同模型同配置的端到端对比 =====")
    print(f"torch({profiler.device}): {torch_ms:.3f}ms/iter")
    print(f"onnxruntime({onnx_prof.providers[0]}): {ort_ms:.3f}ms/iter")
