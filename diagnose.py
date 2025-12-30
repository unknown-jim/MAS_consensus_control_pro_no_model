"""运行环境诊断脚本。

用于快速确认 CUDA / PyTorch 环境是否可用，并通过两组矩阵乘法粗测 GPU 性能。
该脚本只做打印输出，不会修改任何训练配置或文件。

用法:
    python diagnose.py
"""

from __future__ import annotations

import time

import torch


def diagnose() -> None:
    """打印 CUDA 状态与粗粒度性能指标。"""

    print("=" * 60)
    print("🔍 性能诊断")
    print("=" * 60)

    # 1) CUDA 检查
    print("\n1️⃣ CUDA 状态:")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("   ❌ CUDA 不可用：请检查驱动/容器或 PyTorch 安装是否为 CUDA 版本")
        return

    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"   计算能力: {props.major}.{props.minor}")
    print(f"   显存总量: {props.total_memory / 1024**3:.2f} GB")

    # 2) PyTorch 版本
    print(f"\n2️⃣ PyTorch: {torch.__version__}")

    # 3) 简单 GPU 测试（矩阵乘法）
    print("\n3️⃣ GPU 粗测（matmul）:")

    # 小矩阵
    x_small = torch.randn(100, 100, device="cuda")
    start = time.time()
    for _ in range(1000):
        _ = torch.matmul(x_small, x_small)
    torch.cuda.synchronize()
    t_small = time.time() - start
    print(f"   小矩阵 (100x100, 1000 次): {t_small:.3f}s")

    # 大矩阵
    x_large = torch.randn(2000, 2000, device="cuda")
    start = time.time()
    for _ in range(100):
        _ = torch.matmul(x_large, x_large)
    torch.cuda.synchronize()
    t_large = time.time() - start
    print(f"   大矩阵 (2000x2000, 100 次): {t_large:.3f}s")

    # 4) 显存状态
    print(f"\n4️⃣ GPU 显存:")
    print(f"   已分配 (allocated): {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   已保留 (reserved):   {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # 5) 建议
    print("\n" + "=" * 60)
    print("💡 建议:")

    if t_small > 0.5:
        print("   - 小矩阵很慢：可能有其他进程占用 GPU，建议用 `nvidia-smi` 检查")
    if t_large > 1.0:
        print("   - 大矩阵偏慢：可能出现散热/功耗限制（thermal throttling）")

    print("   - 训练时可用 `watch -n 1 nvidia-smi` 观察利用率与显存")
    print("=" * 60)


if __name__ == "__main__":
    diagnose()
