"""Conv+BN融合对照实验：baseline resnet18 vs fx融合版本

融合原理：eval模式下BN是纯线性变换，可折叠进前置卷积的weight/bias，
前向图中BN节点被完全消除。
"""
import time

import torch
import torchvision.models as models
from torch.fx.experimental.optimization import fuse
from torch.profiler import profile, ProfilerActivity, record_function, schedule


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def profile_model(tag: str, model, inputs, device: str, sort_by: str):
    """用schedule采集一个周期，返回该周期内单轮平均耗时(ms)"""
    my_schedule = schedule(skip_first=5, wait=1, warmup=1, active=5, repeat=1)
    num_steps = 5 + (1 + 1 + 5) * 1

    result = {}

    def trace_handler(p: torch.profiler.profile):
        print(f"===== [{tag}] trace ready at step {p.step_num} =====")
        print(p.key_averages().table(sort_by=sort_by, row_limit=8))
        for avg in p.key_averages():
            if avg.key == "model_inference":
                result["avg_ms"] = avg.cpu_time_total / avg.count / 1000

    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else []),
        schedule=my_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
    ) as prof:
        for _ in range(num_steps):
            with record_function("model_inference"):
                with torch.no_grad():
                    model(inputs)
                sync(device)
            prof.step()

    return result.get("avg_ms")


def benchmark(model, inputs, device: str, warmup: int = 10, iters: int = 50):
    """纯wall-clock计时，作为profiler之外的交叉验证"""
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(inputs)
        sync(device)
    return (time.perf_counter() - start) / iters * 1000


if __name__ == "__main__":

    device = get_device()
    print(f"Device: {device}\n")

    inputs = torch.randn(1, 3, 224, 224, device=device)

    baseline = models.resnet18().eval().to(device)
    # fx融合必须在eval模式下进行，否则BN统计量还会更新，折叠不等价
    fused = fuse(baseline)

    # 正确性校验：融合前后输出应数值一致
    with torch.no_grad():
        out_a, out_b = baseline(inputs), fused(inputs)
    max_diff = (out_a - out_b).abs().max().item()
    print(f"融合前后输出最大误差: {max_diff:.2e}")
    assert max_diff < 1e-3, "融合后输出不一致！"

    # 确认BN确实被消除
    n_bn_base = sum(isinstance(m, torch.nn.BatchNorm2d) for m in baseline.modules())
    n_bn_fused = sum(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
    print(f"BatchNorm2d层数: baseline={n_bn_base} -> fused={n_bn_fused}\n")

    sort_by = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"

    prof_base = profile_model("baseline", baseline, inputs, device, sort_by)
    prof_fused = profile_model("fused", fused, inputs, device, sort_by)

    bench_base = benchmark(baseline, inputs, device)
    bench_fused = benchmark(fused, inputs, device)

    print("===== 汇总 =====")
    print(f"{'':>12}  {'profiler avg':>14}  {'wall-clock avg':>14}")
    print(f"{'baseline':>12}  {prof_base:>12.3f}ms  {bench_base:>12.3f}ms")
    print(f"{'fused':>12}  {prof_fused:>12.3f}ms  {bench_fused:>12.3f}ms")
    print(f"{'提升':>12}  {(1 - prof_fused / prof_base) * 100:>12.1f}%  "
          f"{(1 - bench_fused / bench_base) * 100:>12.1f}%")
