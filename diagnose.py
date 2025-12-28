"""
诊断脚本 - 找出性能瓶颈
"""
import torch
import time
import sys

def diagnose():
    print("=" * 60)
    print("🔍 Performance Diagnosis")
    print("=" * 60)
    
    # 1. CUDA 检查
    print("\n1️⃣ CUDA Status:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   Compute capability: {props.major}.{props.minor}")
        print(f"   Total memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("   ❌ CUDA NOT AVAILABLE - This is the problem!")
        return
    
    # 2. PyTorch 版本
    print(f"\n2️⃣ PyTorch: {torch.__version__}")
    
    # 3. 简单 GPU 测试
    print("\n3️⃣ GPU Performance Test:")
    
    # 小矩阵
    x_small = torch.randn(100, 100, device='cuda')
    start = time.time()
    for _ in range(1000):
        y = torch.matmul(x_small, x_small)
    torch.cuda.synchronize()
    t_small = time.time() - start
    print(f"   Small matrix (100x100, 1000x): {t_small:.3f}s")
    
    # 大矩阵
    x_large = torch.randn(2000, 2000, device='cuda')
    start = time.time()
    for _ in range(100):
        y = torch.matmul(x_large, x_large)
    torch.cuda.synchronize()
    t_large = time.time() - start
    print(f"   Large matrix (2000x2000, 100x): {t_large:.3f}s")
    
    # 4. 内存状态
    print(f"\n4️⃣ GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # 5. 建议
    print("\n" + "=" * 60)
    print("💡 Recommendations:")
    
    if t_small > 0.5:
        print("   - GPU seems slow, check nvidia-smi for other processes")
    if t_large > 1.0:
        print("   - Large matrix ops slow, may be thermal throttling")
    
    print("   - Run 'nvidia-smi' to check GPU utilization")
    print("   - Run 'watch -n 1 nvidia-smi' during training")
    print("=" * 60)


if __name__ == '__main__':
    diagnose()