# encoding:utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image


# 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
def loadtraindata(root, datatxt, batch_size):
    print("!!!!!!!!!!inside loadTrainData!!!!!!!!!!!!!!!")
    train_transformations = transforms.Compose([transforms.Resize(227), transforms.CenterCrop(227),
                                                transforms.ToTensor()])
    train_data = MyDataset(root=root, datatxt=datatxt, transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),  # output_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),  # output_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 50)
        )

    def forward(self, x):
        feature_out = self.feature(x)
        f1 = open('./CNNUtil/feature_out_train.txt', 'r+')
        f1.read()
        f1.write(str(feature_out) + "\n")
        f1.close()
        res = feature_out.view(feature_out.size(0), -1)
        f2 = open('./CNNUtil/res_train.txt', 'r+')
        f2.read()
        f2.write(str(res) + "\n")
        f2.close()
        print("res" + str(res.shape))
        out = self.dense(res)
        return out


classes = ('stay', 'left', 'up', 'right', 'down')


def train(savePath, lr, momentum=0.9):
    global trainloader
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 假设我们在支持CUDA的机器上，我们可以打印出CUDA设备：
    net.to(device)
    for epoch in range(3):
        running_loss = 0.0
        print("\n-------------------------------------------\nepoch " + str(epoch) + ":")
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            print(labels)
            # wrap them in Variable
            # inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 39 == 1:
                print('\n[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    torch.save(net, savePath)
    print('Finished Training')


batchSizes = []
learningRates = []


def trainandsave():
    global batchSizes, learningRates, trainloader
    batchSizes.append(4)
    learningRates.append(0.001)
    for batchSize in batchSizes:
        trainloader = loadtraindata("./", "ImagePath_label.txt", batchSize)
        for lr in learningRates:
            savePath = "CNN_" + str(batchSize) + "_" + str(lr) + ".pkl"
            print(savePath)
            if not os.path.exists(savePath):
                os.mknod(savePath)
            train(savePath, lr)


if __name__ == '__main__':
    trainandsave()
