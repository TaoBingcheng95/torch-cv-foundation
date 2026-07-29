"""
硬件相关工具：设备选择与硬件环境快照

- :func:`select_device`          : 智能选取训练设备（auto: cuda → mps → cpu）
- :func:`collect_hardware_report`: 采集可 JSON 序列化的硬件环境快照，用于实验复现
"""

import json
import platform
from typing import Any

import torch


def select_device(name: str = "auto") -> torch.device:
    """
    智能选取训练设备。

    Args:
        name: 设备名。'auto' 按 cuda → mps → cpu 优先级自动选择；
              也可显式指定 'cuda' / 'cuda:1' / 'mps' / 'cpu'（大小写不敏感）。

    Returns:
        torch.device 实例

    Raises:
        ValueError: 显式指定的后端不可用，或 CUDA 设备索引越界
            （fail fast，避免延迟到 model.to() 时才报晦涩错误）
    """
    name = str(name).strip().lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)  # 非法名称由 torch 抛 RuntimeError

    # 显式指定后端时立刻校验可用性
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA requested ('{name}') but not available. "
                f"Available: {_available_backends()}"
            )
        # 校验设备索引（如 'cuda:1' 但只有 1 块卡）
        index = device.index or 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} out of range "
                f"(only {torch.cuda.device_count()} device(s) available)"
            )
    elif device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError(
                f"MPS requested ('{name}') but not available. "
                f"Available: {_available_backends()}"
            )

    return device


def _available_backends() -> list[str]:
    """返回当前环境可用的后端列表（用于报错提示）。"""
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        backends.append("mps")
    return backends


def collect_hardware_report() -> dict[str, Any]:
    """
    Hardware capability reporting for reproducible experiment logs.
    Return a JSON-serializable snapshot of available compute backends.
    usage:
        print(json.dumps(collect_hardware_report(), indent=2, sort_keys=True))
    """
    # opencv 为可选依赖，仅用于版本记录，惰性导入避免绑架设备选择主流程
    try:
        import cv2
        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = None

    cuda_available = torch.cuda.is_available()
    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0] if mps_available else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": opencv_version,
        "cuda": {
            "available": bool(cuda_available),
            "version": torch.version.cuda,  # torch 编译所用 CUDA 版本，无 CUDA 时为 None
            "cudnn": torch.backends.cudnn.version() if cuda_available else None,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "devices": cuda_devices,
        },
        "mps": {
            "available": mps_available,
            "built": bool(mps_backend is not None and mps_backend.is_built()),
        },
    }



if __name__ == "__main__":
    print(json.dumps(collect_hardware_report(), indent=2, sort_keys=True))



