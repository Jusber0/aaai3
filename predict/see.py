import torch

# 假设文件名为 'model.pth'
model_path = 'model_params.pth'

# 加载模型权重
model = torch.load(model_path)

# 打印模型的内容（例如：参数字典）
print(model)
