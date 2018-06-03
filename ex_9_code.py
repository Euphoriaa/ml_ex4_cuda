import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

#hyper-params
########################
#optimizer
lr = 0.01
momentum = 0.7

#Channels, filter sizes
conv1_channels = 150
conv2_channels = 100
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_channels, 2)
        self.pad1 = nn.ConstantPad2d(1,0)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnormconv1 = nn.BatchNorm2d(conv1_channels)        
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 2)
        self.batchnormconv2 = nn.BatchNorm2d(conv2_channels)    
        self.fc1 = nn.Linear(conv2_channels * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 84)
        self.batchnorm2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pad1(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batchnormconv1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batchnormconv2(x)
        #TODO(david): remove this assertion
        #length = self.num_flat_features(x)        
        #assert(length == 16 * 7 * 7)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    #Using GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    print(device)

    net = Net()
    net.to(device)

    #handling validation and training split
    dataset_size = 50000  # from documentation of data set
    training_size = int(dataset_size*0.8)
    validation_size = dataset_size - training_size

    indices = list(range(dataset_size))
    split = validation_size
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_idx)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(#I calculated seperately
                                                         [0.49139969, 0.48215842, 0.44653093], #Mean per channel
                                                         [0.06102493, 0.05928501, 0.0684282])])#STD per channel

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              num_workers=2, sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              num_workers=2, sampler=validation_sampler)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             num_workers=2)

    classes = ('ariplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)#, momentum=momentum)
    
    loss_train = []
    loss_valid = []
    acc_train = []
    acc_validation = []
    epochs = range(1,16)
    for epoch in epochs:  # loop over the dataset multiple times
        #training
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        
        #check acc and loss on validation and training
        with torch.no_grad():
            correct = 0
            total = 0
            loss_sum = 0
            
            #testing on training
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss_sum += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct+= (predicted == labels).sum().item()
            
            #saving loss and acc
            loss_train.append(loss_sum / training_size)
            acc = 100 * correct / total
            acc_train.append(acc)
            print('Accuracy on the %d training images : %d %%' % (total,
                acc))
            
            #copy-paste for validation
            correct = 0
            total = 0
            loss_sum = 0
            for i, data in enumerate(validloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss_sum += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct+= (predicted == labels).sum().item()
            loss_valid.append(loss_sum / validation_size)
            acc = 100 * correct / total
            acc_validation.append(acc)
            print('Accuracy on the %d validation images : %d %%' % (total,
                acc))

    print('Finished Training')

    #check test accuracy
    correct = 0
    total = 0
    loss_sum = 0
    predictions = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss_sum += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            for prediction in predicted:
                predictions.append(prediction.item())
            total += labels.size(0)
            correct+= (predicted == labels).sum().item()
    loss_test = loss_sum / (10000)
    print('Accuracy on the %d test images : %d %%' % (total,
        100 * correct / total))
    result = input('Do you want to save predictions? y/n')
    if result == 'y':
        with open('test.pred','w') as f:
            for prediction in predictions:
                f.write(str(prediction) + '\n')
    loss_avg_train = [x / training_size for x in loss_train ]
    loss_avg_valid = [x / validation_size for x in loss_valid ]
    #plot average loss per epoch for the validaion and training set
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.plot(epochs, loss_avg_train)
    plt.plot(epochs, loss_avg_valid)
    plt.legend(('Average training loss','Average validation loss'))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    print('Train avg loss: {}'.format(loss_train[-1]))
    print('Validation avg loss: {}'.format(loss_valid[-1]))
    print('Test avg loss: {}'.format(loss_test))
