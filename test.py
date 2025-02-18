import pytorch_lightning as pl
from torch import optim, nn, utils, Tensor
from model import TransformerModel_no_softmax
from loss import edl_digamma_loss
from dataset import prepare_data,tabular_transformer
from torch.utils.data import DataLoader
import torch
from torchmetrics import Accuracy, F1Score, AUROC
from utils import one_hot_embedding, relu_evidence

# 加载训练好的模型
model = TransformerModel_no_softmax(ip_dim=7, seq_len=15, d_model=32, 
                                    nhead=2, d_hid=32, nlayers=2, dropout=0.1)

# 加载参数
model.load_state_dict(torch.load("model_params.pth"))
model.eval()  # 将模型设置为评估模式

# 摄像头数据预处理
real_time_data = 0
def preprocess_data(real_time_data):
    pass


# 假设你从摄像头或传感器获取数据 x
# x 的形状需要与训练时的输入格式一致
x = preprocess_data(real_time_data)

# 转换为Tensor，确保形状为 (seq_len, batch_size, ip_dim)
x_tensor = torch.tensor(x).float()

# 进行预测
with torch.no_grad():
    predictions = model(x_tensor)
