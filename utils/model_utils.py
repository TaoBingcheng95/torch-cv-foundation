import time
import torch

__all__ = ['model_summary', 'platform_performance', 'memory_bandwidth']


def model_summary(model_name:str, model:torch.nn.Module, ):
    total_count = 0
    total_byte_count = 0
    for param in model.parameters():
        total_count += param.nelement()
        total_byte_count += param.nelement()*param.element_size()
    # buffer (如 BN 的 running_mean/var) 同样占用内存
    buffer_byte_count = 0
    for buf in model.buffers():
        buffer_byte_count += buf.nelement()*buf.element_size()
    print('=======================================')
    print(f'Model: {model_name}')
    print(f'Number of parameters: {total_count}.')
    print(f'Memory usage (params): {total_byte_count/1024/1024:.4f} MB')
    print(f'Memory usage (buffers): {buffer_byte_count/1024/1024:.4f} MB')
    print('=======================================')


def _get_device() -> torch.device:
    """按 cuda > mps > cpu 优先级选择设备。"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _synchronize(device: torch.device):
    """等待设备上所有异步 kernel 执行完毕。"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()



def platform_performance(size=10000, warmup=5, iters=10, dtype=torch.float32, device=None):
    """
    平台性能测试: 用 size x size 矩阵乘法测量设备实际算力 (TFLOPS)。

    Args:
        size: 矩阵边长, 矩阵乘 FLOPs = 2 * size^3
        warmup: 预热次数, 排除首次 kernel 编译/缓存分配的开销
        iters: 计时迭代次数
        dtype: 测试精度, 如 torch.float32 / torch.float16
        device: 指定设备, None 时按 cuda > mps > cpu 自动选择

    Returns:
        dict: 包含平均/最小耗时与 TFLOPS 的测试结果
    """
    print(f"PyTorch 版本: {torch.__version__}")

    device = torch.device(device) if device is not None else _get_device()
    if device.type == 'cuda':
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        # 检查支持的架构 (关键!)
        print(f"支持的架构: {torch.cuda.get_arch_list()}")
        # TF32 会影响 fp32 matmul 结果, 明确打印当前开关状态
        print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    elif device.type == 'mps':
        print(f"MPS 可用: {torch.backends.mps.is_available()}")
    else:
        print(f"CPU 线程数: {torch.get_num_threads()}")
    print(f"设备: {device}, 精度: {dtype}, 矩阵: {size}x{size}")

    # 矩阵乘法测试
    x = torch.randn(size, size, device=device, dtype=dtype)
    y = torch.randn(size, size, device=device, dtype=dtype)

    # 预热
    for _ in range(warmup):
        _ = x @ y
    # 预热的 kernel 是异步提交的, 必须先同步, 否则会被算进计时段
    _synchronize(device)

    # 逐次计时, 便于观察波动 (每次都同步, 测的是单次完整耗时)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = x @ y
        _synchronize(device)
        times.append(time.perf_counter() - start)

    flops = 2 * size ** 3  # 单次矩阵乘的浮点运算数
    avg_t = sum(times) / len(times)
    min_t = min(times)
    result = {
        'device': str(device),
        'dtype': str(dtype),
        'size': size,
        'avg_time_s': avg_t,
        'min_time_s': min_t,
        'avg_tflops': flops / avg_t / 1e12,
        'peak_tflops': flops / min_t / 1e12,
    }

    print(f"{iters}次 {size}x{size} 矩阵乘法总耗时: {sum(times):.3f} 秒")
    print(f"平均每次: {avg_t*1000:.1f} ms, 最快: {min_t*1000:.1f} ms")
    print(f"实测算力: 平均 {result['avg_tflops']:.2f} TFLOPS, 峰值 {result['peak_tflops']:.2f} TFLOPS")
    return result


def memory_bandwidth(size_mb=1024, warmup=5, iters=10, device=None):
    """
    显存/内存带宽测试: 用大张量拷贝测量设备带宽 (GB/s)。
    """
    device = torch.device(device) if device is not None else _get_device()
    n = size_mb * 1024 * 1024 // 4  # float32 元素个数
    x = torch.randn(n, device=device)

    for _ in range(warmup):
        _ = x.clone()
    _synchronize(device)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = x.clone()
        _synchronize(device)
        times.append(time.perf_counter() - start)

    # clone = 读一次 + 写一次
    bytes_moved = 2 * n * 4
    bw = bytes_moved / min(times) / 1e9
    print(f"设备: {device}, 拷贝 {size_mb} MB, 实测带宽: {bw:.1f} GB/s")
    return bw


if __name__ == "__main__":
    platform_performance()
    memory_bandwidth()

