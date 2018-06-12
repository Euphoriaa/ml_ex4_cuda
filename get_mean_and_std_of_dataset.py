import torch

import torchvision
import torchvision.transforms as transforms
import numpy as np

def print_mean_std(loader):
    with torch.no_grad():
        example = iter(loader).next()
        #normalizing factor of 1024
        factor = example[0].shape[2]*example[0].shape[3]
        _sum = np.array([0., 0., 0.])
        _total = 0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            _total += labels.size(0)
            input_sum = np.array([torch.sum(x).item() for x in torch.sum(inputs,0)])
            _sum += input_sum
        _avg = _sum / (_total*factor)
        print('Mean: ' + str(_avg))
        #std
        _std_sum = np.array([0.,0.,0.])
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            for example in inputs:
                minus_avg = [(x.sub_(avg)).pow(2) for (x,avg) in zip(example,_avg)]
                sums = []
                for channel in minus_avg:
                    sums.append(channel.sum().item())
                for i,__sum in enumerate(sums):
                    _std_sum[i] += __sum

        _std = _std_sum / (_total*factor)
        print('STD: ' + str(_std))
if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              num_workers=2, shuffle=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             num_workers=2)
    with torch.no_grad():
        train_sum = np.array([0, 0, 0])
        train_total = 0
        print('Training stats:')
        print_mean_std(trainloader)
        print('Test stats:')
        print_mean_std(testloader)
        '''
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            train_total += labels.size(0)
            input_sum = np.array([torch.sum(x).item() for x in torch.sum(inputs,0)])
            train_sum += input_sum
        train_avg = train_sum / train_total
        print(str(train_avg))
        train_std_sum = np.array([0,0,0])
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs.sub_(train_avg)
            inputs.pow_(2)
            input_sum = np.array([torch.sum(x).item() for x in torch.sum(inputs, 0)])
            train_std_sum += input_sum
        train_std = train_std_sum / train_total
        print(str(train_std))

        test_sum = torch.zeros(3,1)
        test_total = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            test_total += labels.size(0)
            input_sum = np.array([torch.sum(x).item() for x in torch.sum(inputs, 0)])
            test_sum += input_sum
        test_avg = test_sum / test_total
        '''