#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from utils import progress_bar
import argparse
import os

import googlenet

show = ToPILImage() #可以把Tensor转成Image，方便可视化

class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 100)  
        self.fc2   = nn.Linear(100, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x
    
class lenet_mnist(nn.Module):
    def __init__(self):
        super(lenet_mnist, self).__init__()
        self.cnn = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2)
        ) 
        self.fullyConnections = nn.Sequential(
        nn.Linear(16 * 5 * 5, 120),
        nn.Linear(120, 84),
        nn.Linear(84, 10)
        ) 
    def forward(self, x): 
        x = self.cnn(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fullyConnections(x) 
        return x
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', '-d', default='cifar10', help='chose a dataset cifar10 or mnist')
args = parser.parse_args()


#data_set = 'cifar10'
#data_set = 'mnist'
data_set = args.dataset
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#device = 'cuda' if t.cuda.is_available() else 'cpu'
#device = 'cpu'

print("Use "+ str(device))

#创建一个网络
if(data_set =='cifar10'):
    net = lenet()
    #net = googlenet.GoogLeNet()
elif(data_set == 'mnist'):
    net = lenet_mnist()
    #net = lenet()


net = net.to(device)

if device == 'cuda:0':
    net = t.nn.DataParallel(net)
    cudnn.benchmark = True
    print("cuda is ready")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = t.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
        
        
# Data
print('==> Preparing data..')
# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if(data_set == 'cifar10'):
    # 训练集
    print('Use CIFAR-10 Data')
    trainset = tv.datasets.CIFAR10(
                        root='/home/pakcy/Downloads/pytorch-cifar-master/data', 
                        train=True, 
                        download=True,
                        transform=transform)

    trainloader = t.utils.data.DataLoader(
                        trainset, 
                        batch_size=128,
                        shuffle=True, 
                        num_workers=2)

    # 测试集
    testset = tv.datasets.CIFAR10(
                        '/home/pakcy/Downloads/pytorch-cifar-master/data',
                        train=False, 
                        download=True, 
                        transform=transform)

    testloader = t.utils.data.DataLoader(
                        testset,
                        batch_size=4, 
                        shuffle=False,
                        num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
elif(data_set == 'mnist'):
    # 训练集
    print('Use MNIST Data')
    trainset = tv.datasets.MNIST(
                        root='mnist/', 
                        train=True, 
                        download=True,
                        transform=transform_train)

    trainloader = t.utils.data.DataLoader(
                        trainset, 
                        batch_size=128,
                        shuffle=True, 
                        num_workers=2)

    # 测试集
    testset = tv.datasets.MNIST(
                        root='mnist/',
                        train=False, 
                        download=True, 
                        transform=transform_train)

    testloader = t.utils.data.DataLoader(
                        testset,
                        batch_size=4, 
                        shuffle=False,
                        num_workers=2)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')

    

#train 
t.set_num_threads(8)
def train(epoch):
    
    running_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 0):
        
        # 输入数据
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 梯度清零
        # 在训练过程中
        # 先梯度清零(与net.zero_grad()效果一样)
        optimizer.zero_grad()
        
        # forward + backward 
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        #反向传播
        loss.backward()   
        
        # 更新参数 
        optimizer.step()
        
        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f'
        % (running_loss/(batch_idx+1)))
        #if batch_idx % 20 == 0: # 每2000个batch打印一下训练状态
            #print('[%d, %5d] loss: %.3f' \
             #     % (epoch+1, batch_idx+1, running_loss / 20))
            #running_loss = 0.0

def test(epoch):
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with t.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
        
    
def test2(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with t.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    #print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        t.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

        
if __name__ == '__main__':   
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #新建一个优化器，指定要调整的参数和学习率
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))
    
    
    dataiter = iter(testloader)
    train_epoch = 200
    
    for epoch in range(start_epoch, start_epoch+train_epoch):
        train(epoch)
        #test(epoch)
        test2(epoch)
        print("Epoch : " + str(epoch+1))
        print('Best acc :' + str(best_acc))
        images, labels = dataiter.next() # 一个batch返回4张图片
        show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100)).show()
        images, labels = images.to(device), labels.to(device)
        print('实际的label: ', ' '.join(\
                                     '%08s'%classes[labels[j]] for j in range(4)))
        
        #计算图片在每个类别上的分数
        outputs = net(images)
        # 得分最高的那个类
        _, predicted = t.max(outputs.data, 1)
        print('预测lable: ', ' '.join('%5s'\
                                    % classes[predicted[j]] for j in range(4)))
    print('Finished Training')
    

