import torch
import time

# 1. 检查版本
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda}")

# 2. 检查支持的架构 (关键!)
print(f"支持的架构: {torch.cuda.get_arch_list()}")

# 3. 检查 GPU 是否可用
print(f"GPU 可用: {torch.cuda.is_available()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

# 4. 简单计算测试
x = torch.randn(1000, 1000).cuda()
y = x @ x
print(f"计算测试通过! 结果形状: {y.shape}")



device = torch.device('cuda')
size = 10000

# 矩阵乘法测试
x = torch.randn(size, size, device=device)
y = torch.randn(size, size, device=device)

# 预热
_ = x @ y

# 计时
start = time.time()
for _ in range(10):
    z = x @ y
torch.cuda.synchronize()
end = time.time()

print(f"RTX 5070 Ti 性能测试:")
print(f"10次 {size}x{size} 矩阵乘法耗时: {end - start:.3f} 秒")
print(f"平均每次: {(end - start) / 10:.3f} 秒")

