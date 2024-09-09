import torch.nn as nn
import torchvision

class MyResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(MyResNet, self).__init__()
        self.res34 = torchvision.models.resnet34(pretrained=True)
        self.res34.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res34(x)
        return x
    
