import torch
# 查看pytorch版本
print(f'pytorch版本: {torch.version.__version__}')
# 查看显卡GPU是否可用
print(f'GPU是否可用: {torch.cuda.is_available()}')
# 查看GPU可用数
print(f'GPU可用数: {torch.cuda.device_count()}')
# 查看CUDA版本
print(f'CUDA版本: {torch.version.cuda}')
# 查看CUDA-cuDNN版本
print(f'cuDNN版本: {torch.backends.cudnn.version()}')
quit()
