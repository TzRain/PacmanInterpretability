import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models

device = torch.device("cpu")


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        net = models.alexnet(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.rnn = nn.LSTM(input_size=256 * 6 * 6, hidden_size=64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, 5)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 3, 224, 224)
        output = self.features(x)
        output = output.view(B, -1).transpose(0, 1).contiguous().view(256 * 6 * 6, B, 1)
        output = output.permute(1, 2, 0)
        h0 = torch.zeros(2, x.size(0), 64).to(device)
        c0 = torch.zeros(2, x.size(0), 64).to(device)
        # 前向传播LSTM$
        out, _ = self.rnn(output, (h0, c0))  # 输出大小 (batch_size, seq_length, hidden_size)
        out = self.out(out[:, -1, :])
        return out


class RNNHtMp:
    def __init__(self, path='../ModelTrainer/RNN_24_0.001.pth'):
        self.path = path
        self.model = ConvLSTM()
        self.model.load_state_dict(torch.load(path))
        self.transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

    def setModel(self, path):
        self.path = path
        self.model.load_state_dict(torch.load(path))

    def draw_CAM(self, img_path):
        # 图像加载&预处理
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = img.unsqueeze(0)
        # 获取模型输出的feature/score
        self.model.eval()
        feature_1 = self.model.features.features(img)
        feature_2 = feature_1.view(1, -1).transpose(0, 1).contiguous().view(256 * 6 * 6, 1, 1)
        feature_2 = feature_2.permute(1, 2, 0)
        h0 = torch.zeros(2, img.size(0), 64).to(device)
        c0 = torch.zeros(2, img.size(0), 64).to(device)
        # 前向传播LSTM$
        features, _ = self.model.rnn(feature_2, (h0, c0))
        output = self.model.out(features)

        # 为了能读取到中间梯度定义的辅助函数
        def extract(g):
            global features_grad
            features_grad = g

        # 预测得分最高的那一类对应的输出score
        pred = torch.argmax(output).item()
        pred_class = output[0][0][pred]
        feature_1.register_hook(extract)
        pred_class.backward()
        grads = features_grad  # 获取梯度
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
        pooled_grads = pooled_grads[0]
        feature_1 = feature_1[0]
        feature_1 = feature_1.permute(1, 2, 0)  # 变成（6，6，256）
        for i in range(256):
            feature_1[:, :, i] *= pooled_grads[i]
        heatmap = feature_1.detach().numpy()
        heatmap = np.mean(heatmap, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                if heatmap[i][j] == 1:
                    return i, j
