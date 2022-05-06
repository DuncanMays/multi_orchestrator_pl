import torch
F = torch.nn.functional


class TwoNN(torch.nn.Module):

    def __init__(self):
        super(TwoNN, self).__init__()
        
        self.fc1 = torch.nn.Linear(784, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

class ThreeNN(torch.nn.Module):

    def __init__(self):
        super(ThreeNN, self).__init__()
        
        self.fc1 = torch.nn.Linear(784, 500)
        self.fc2 = torch.nn.Linear(500, 300)
        self.fc3 = torch.nn.Linear(300, 100)
        self.fc4 = torch.nn.Linear(100, 10)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x

# the convolutional network used in FedAvg paper
class ConvNet(torch.nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, (5, 5))
        self.mp1 = torch.nn.MaxPool2d((2,2))
        self.conv2 = torch.nn.Conv2d(32, 64, (5, 5))
        self.mp2 = torch.nn.MaxPool2d((2,2))

        self.flatten = torch.nn.Flatten()
        
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.mp2(x)

        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x