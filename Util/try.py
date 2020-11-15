import pickle
import HeatMapCNN as CHMG
import HeatMapRNN as RHMG
import torch as t
from torch import nn
import math

device = t.device('cpu')


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


def transfer(p, sz):
    x, y = p
    n, m = sz
    hx = (x + 1 / 2) * n / 6
    hy = (y + 1 / 2) * m / 6
    return hx, hy


def dis(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == '__main__':
    pkl_file = open("../result/trainListWithEyePos.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    rnnHtMp = RHMG.RNNHtMp("../models/RNN_4_0.001.pth")
    cnnHtMp = CHMG.CNNHtMp()

    dataList = data["dataList"]
    i = 0
    for item in dataList:
        # if i >= 100:
        #     break
        print(i)
        if i >= data["eyePosLength"]:
            break
        path = item["path"]
        CNNArea = (-1, -1)
        RNNArea = (-1, -1)
        CNNPoint = (-1, -1)
        RNNPoint = (-1, -1)
        CNNDis = 1000
        RNNDis = 1000

        sz = item["eyePos"][-2:]
        eye = item["eyePos"][:2]

        try:
            CNNArea = cnnHtMp.draw_CAM(path)
            CNNPoint = transfer(CNNArea, sz)
            CNNDis = dis(CNNPoint, eye)
        except Exception as err:
            print("CNN")
            print(err)

        try:
            RNNArea = rnnHtMp.draw_CAM(path)
            RNNPoint = transfer(RNNArea, sz)
            RNNDis = dis(RNNPoint, eye)
        except Exception as err:
            print("RNN")
            print(err)

        item["CNNArea"] = CNNPoint
        item["RNNArea"] = RNNPoint
        item["CNNPoint"] = CNNPoint
        item["RNNPoint"] = RNNPoint
        item["CNNDis"] = CNNDis
        item["RNNDis"] = RNNDis

        i += 1

    pkl_file = open("../result/trainListWithEyePosWithHeatMap.pkl", 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close()
