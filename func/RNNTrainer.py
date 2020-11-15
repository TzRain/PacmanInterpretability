import os
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models

device = torch.device('cpu')

dic = {'stay': 0, 'left': 1, 'up': 2, 'right': 3, 'down': 4}


# 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
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
        fn, label = self.imgs[index]
        # fn是图片path
        # #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
def loadTrainData(root, datatxt, batch_size=4):
    # root = r"../../data/marui/dataset/train/"
    train_transformations = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224),
                                                transforms.ToTensor()])
    train_data = MyDataset(root=root, datatxt=datatxt, transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader


def loadTestData(root, datatxt, batch_size=4):
    # root = r"../../data/marui/dataset/test/"
    test_transformations = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224),
                                               transforms.ToTensor()])
    test_data = MyDataset(root=root, datatxt=datatxt, transform=test_transformations)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        net = models.alexnet(pretrained=True)
        print(net)
        net.classifier = nn.Sequential()
        self.features = net
        print('featrues:', self.features)
        self.rnn = nn.LSTM(input_size=256 * 6 * 6, hidden_size=64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, 5)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 3, 224, 224)

        output = self.features(x)
        print('0:', output.shape)
        output = output.view(B, -1).transpose(0, 1).contiguous().view(256 * 6 * 6, B, 1)
        output = output.permute(1, 2, 0)

        h0 = torch.zeros(2, x.size(0), 64).to(device)
        c0 = torch.zeros(2, x.size(0), 64).to(device)

        # 前向传播LSTM
        print('brefore', output.shape)
        out, _ = self.rnn(output, (h0, c0))  # 输出大小 (batch_size, seq_length, hidden_size)
        # 解码最后一个时刻的隐状态
        out = self.out(out[:, -1, :])
        return out


model = ConvLSTM().to(device)
learningRates = []
batchSizes = []


# 训练
def train(savePath, learningRate=0.001, momentum=0.9):
    global model, train_loader, test_loader
    loss_plt = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    print('begin training...')
    current_largest = 50000
    print('model created.\n')
    for epoch in range(3):
        running_loss = 0.0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            print(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
        print('epoch:', epoch, 'running loss:', running_loss / total)
        # if epoch % 5 == 0:
        if running_loss < current_largest:
            print('save_epoch:', epoch, 'running loss:', running_loss / total)
            current_largest = running_loss
            torch.save(model.state_dict(), savePath)
        final_loss = running_loss / total
        loss_plt.append(final_loss)


# 测试
def test():
    global model, train_loader, test_loader
    print('begin testing')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.int()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))


def creatModels():
    global model, train_loader, test_loader, learningRates, batchSizes
    learningRates.append(0.001)
    batchSizes.append(4)
    root = "./"
    datatxt = "ImagePath_label_train.txt"
    for batchSize in batchSizes:
        train_loader = loadTrainData(root, datatxt, batchSize)
        test_loader = loadTestData(root, datatxt, batchSize)
        for learningRate in learningRates:
            savePath = "RNN_" + str(batchSize) + "_" + str(learningRate) + ".pth"
            print(savePath)
            if not os.path.exists(savePath):
                os.mknod(savePath)
            train(savePath, learningRate)


if __name__ == '__main__':
    creatModels()
