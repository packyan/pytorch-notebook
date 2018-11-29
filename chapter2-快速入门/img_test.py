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
from torchvision.datasets import ImageFolder
import googlenet
import cv2
from skimage import io
show = ToPILImage() #可以把Tensor转成Image，方便可视化

class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 100)  
        self.fc2   = nn.Linear(100, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        preoout = []
        x = self.conv1(x)
        preoout.append(x)

        x = F.max_pool2d(F.relu(x), (2, 2)) 
        preoout.append(x)
        
        x = self.conv2(x)
        preoout.append(x)

        x = F.max_pool2d(F.relu(x), 2) 
        preoout.append(x)
        
        x = x.view(x.size()[0], -1) 
        preoout.append(x)
        
        x = F.relu(self.fc1(x))
        preoout.append(x)
        
        x = F.relu(self.fc2(x))
        preoout.append(x)
        
        x = self.fc3(x) 
        preoout.append(x)
        
        return x,preoout
    
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
        preoout = []
        preoout.append(x)
        x = self.cnn(x)
        preoout.append(x)
        x = x.view(-1, 16 * 5 * 5)
        preoout.append(x)
        x = self.fullyConnections(x)
        preoout.append(x)
        return x,preoout

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testimg', default='2.png', type=str, help='test image name')
args = parser.parse_args()


img_to_test = '1.'
#data_set = 'cifar10'
data_set = 'mnist'

#device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#device = 'cuda' if t.cuda.is_available() else 'cpu'
device = 'cpu'

print("Use "+ str(device))

#创建一个网络
if(data_set =='cifar10'):
    net = lenet()
    #net = googlenet.GoogLeNet()
elif(data_set == 'mnist'):
    #net = lenet_mnist()
    net = lenet()

net = net.to(device)

if device == 'cuda:0':
    net = t.nn.DataParallel(net)
    cudnn.benchmark = True
    print("cuda is ready")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = t.load('./checkpoint/ckpt_mnist.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
        
        
# Data
print('==> Preparing data..')
# 定义对数据的预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')
    
best_acc = 0
start_epoch = 0
if __name__ == '__main__':   
    #dataset = ImageFolder('img_test/',transform=transform_train)
    #print(dataset.imgs)
#     mnist_test= tv.datasets.MNIST(
#     './mnist', train=False, download=True)
#     f=open('mnist_test.txt','w')
#     for i,(img,label) in enumerate(mnist_test):
#         #img_path="./img_test/"+str(i)+".jpg"
#         img_path="./img_test/"+str(i)+".jpg"
#         img.save(img_path)
#         f.write(img_path+' '+str(label)+'\n')
# f.close()
    img = cv2.imread(args.testimg)
    print(type(img))
    img2 = transforms.ToPILImage()(img).convert('L')
    img2.show()
    img_tensor = transform_train(img2)
    img_tensor = img_tensor.unsqueeze(0)
    #print(img_tensor.size())
    #计算图片在每个类别上的分数
    outputs = net(img_tensor)
    output = outputs[0]
    for data in outputs[1]:
        print(data.size())
    # 得分最高的那个类
    _, predicted = t.max(output.data, 1)
    print('预测lable: ', ' '.join('%5s'\
                                % classes[predicted[j]] for j in range(1)))

   


