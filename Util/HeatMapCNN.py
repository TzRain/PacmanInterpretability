import numpy as np
import torch
import torch as t
from PIL import Image
from torch import nn
from torchvision import transforms

device = torch.device('cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=3, stride=2),  # output_size = 27*27*96
            t.nn.Conv2d(96, 256, 5, 1, 2),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2),  # output_size = 13*13*256
            t.nn.Conv2d(256, 384, 3, 1, 1),
            t.nn.ReLU(),  # output_size = 13*13*384
            t.nn.Conv2d(384, 256, 3, 1, 1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )

        self.dense = t.nn.Sequential(
            t.nn.Linear(6400, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 50)
        )

    def forward(self, x):
        feature_out = self.feature(x)
        res = feature_out.view(feature_out.size(0), -1)
        out = self.dense(res)
        return out


class CNNHtMp:
    def __init__(self, path="../models/CNN_4_0.001.pkl"):
        self.path = path
        self.model = t.load(path)
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
        feature_1 = self.model.feature(img)
        res = feature_1.view(feature_1.size(0), -1)
        output = self.model.dense(res)

        # 为了能读取到中间梯度定义的辅助函数
        def extract(g):
            global features_grad
            features_grad = g

        # 预测得分最高的那一类对应的输出score
        pred = torch.argmax(output).item()
        pred_class = output[0][pred]
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



