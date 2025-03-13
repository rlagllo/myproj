import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import skimage.transform
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)
        return out


class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10):
        super(ResNet, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # feature map size = 32x32x16
        self.layers_2n = self.get_layers(block, 16, 16, stride=1)
        # feature map size = 16x16x32
        self.layers_4n = self.get_layers(block, 16, 32, stride=2)
        # feature map size = 8x8x64
        self.layers_6n = self.get_layers(block, 32, 64, stride=2)

        # output layers
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, in_channels, out_channels, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, stride, down_sample)]) # block은 residualblock 클래스로 입력받을 거라서, residualblock 정의 보면 됨

        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x) # 64, 3, 32, 32 -> 64, 16, 32, 32
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers_2n(x) # 64, 16, 32, 32
        x = self.layers_4n(x) # 64, 32, 16, 16
        x = self.layers_6n(x) # 64, 64, 8, 8

        feature = x.clone()
        x = self.avg_pool(x)
        #print(f"x.shape: {x.shape}") # 64, 64, 1, 1
        
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x, feature


def resnet():
    block = ResidualBlock
    model = ResNet(5, block)
    return model


trainx_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\train_x.npz'
trainy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\train_y.npz'

validx_path =r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\valid_x.npz'
validy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\valid_y.npz'

testx_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_x.npz'
testy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_y.npz'
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

class mydataset(Dataset):
    def __init__(self, data_path,label_path, transform ): # ,original_transform 추가 가능
        data = np.load(data_path)
        labels = np.load(label_path)

        if "train_x" in data and "train_y" in labels:
            self.data = data["train_x"]
            self.labels = labels["train_y"]
        elif "valid_x" in data and "valid_y" in labels:
            self.data = data["valid_x"]
            self.labels = labels["valid_y"]
        elif "test_x" in data and "test_y" in labels:
            self.data = data["test_x"]
            self.labels = labels["test_y"]
        self.transform = transform
        #self.original_transform = original_transform

    def __len__(self): # dataloader는 __len__함수를 통해 데이터 길이를 확인하고, 그에 따라 idx리스트를 만들어냄
        return (len(self.data))

    def __getitem__(self,idx): # dataloader가 데이터를 요청할 때 자동으로 호출됨 idx변수는 dataloader가 만든거로 줌
        x = self.data[idx]
        y = self.labels[idx]

        # 정의해놓은 transform의 내용을 보면, torchvision의 것을 쓰고있음. 지금 transforms함수가 torchvision의 것인데, torchvision은 지금 현재 데이터 형식에는 안맞아서, 지원이 되는 PIL형식으로 바꾸는 것
        x = Image.fromarray(np.transpose(x, (1, 2, 0)).astype(np.uint8))
        #original = self.original_transform(x)


        if self.transform:
            x = self.transform(x)

        return x, torch.tensor(y,dtype=torch.long)
        # return original, x, torch.tensor(y,dtype=torch.long)



train_dataset = mydataset(trainx_path, trainy_path,transforms_train) # ,original_transform=original_transform 추가 가능
train_loader = DataLoader(train_dataset, shuffle=True,batch_size=64)
valid_dataset = mydataset(validx_path,validy_path,transforms_test) # ,original_transform=original_transform 추가 가능
valid_loader = DataLoader(valid_dataset,shuffle=True,batch_size=64)

net = resnet()
net = net.to('cuda')
#num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

# if args.resume is not None:
#     checkpoint = torch.load('./save_model/' + args.resume)
#     net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)

decay_epoch = [32000, 48000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)
#writer = SummaryWriter(args.logdir)


def train(epoch, global_steps):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    trainloss_list = []
    valid_loss_list = []
    valid_accu_list = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        global_steps += 1  
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_lr_scheduler.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    trainloss_list.append(train_loss / len(train_loader))
    net.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            classification, _ = net(images)
            loss = criterion(classification, labels)
            valid_loss += loss.item()
            _,predicted = torch.max(classification,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_loss_list.append(avg_valid_loss)
        
        valid_accu = 100 * correct / total
        valid_accu_list.append(valid_accu)

    acc = 100 * correct / total
    print('train epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
        epoch, batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc))
    print(f"Epoch [{epoch}] | "
    f"Valid Loss: {avg_valid_loss:.4f} | "
    f"Valid Accuracy: {valid_accu:.2f}%")
    #writer.add_scalar('log/train error', 100 - acc, global_steps)
    return global_steps, trainloss_list


# def test(epoch, best_acc, global_steps):
#     net.eval()

#     test_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs = inputs.to('cuda')
#             targets = targets.to('cuda')
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     acc = 100 * correct / total
#     print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
#         epoch, batch_idx, len(test_loader), test_loss / (batch_idx + 1), acc))

#     writer.add_scalar('log/test error', 100 - acc, global_steps)

#     if acc > best_acc:
#         print('==> Saving model..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('save_model'):
#             os.mkdir('save_model')
#         torch.save(state, './save_model/ckpt.pth')
#         best_acc = acc

#     return best_acc


if __name__ == '__main__':
    best_acc = 0
    epoch = 0
    global_steps = 0

    # if args.resume is not None:
    #     test(epoch=0, best_acc=0)
    #else:
    
    while True:
        epoch += 1
        global_steps, trainlosslist = train(epoch, global_steps)
        #best_acc = test(epoch, best_acc, global_steps)
        #print('best test accuracy is ', best_acc)

        if global_steps >= 64000:
            break
    torch.save(net.state_dict(), "C:\\Users\\김해창\\Desktop\\cifar-10-batches-py\\model_resnet.pth")    
        
        
    plt.plot(trainlosslist, label="Training Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()