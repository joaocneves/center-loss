import torch
from torch import nn
from torchvision.models import resnet18, resnet50
from torch.nn import functional as F

from .base import FaceModel
from device import device


class ConvNet(nn.Module):

    IMAGE_SHAPE = (28, 28)

    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.register_buffer('centers', (
                torch.rand(num_classes, 2).to(device) - 0.5) * 2)

        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        feature = self.prelu_fc1(self.fc1(x))
        logits = self.fc2(feature)

        feature_normed = feature#.div(
        #    torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return logits, feature_normed

class ResnetFaceModel(FaceModel):

    IMAGE_SHAPE = (96, 128)

    def __init__(self, num_classes, feature_dim):
        super().__init__(num_classes, feature_dim)

        self.extract_feature = nn.Linear(
            self.feature_dim*4*3, self.feature_dim)
        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = x.view(x.size(0), -1)
        feature = self.extract_feature(x)
        logits = self.classifier(feature) if self.num_classes else None

        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return logits, feature_normed


class Resnet18FaceModel(ResnetFaceModel):

    FEATURE_DIM = 512

    def __init__(self, num_classes):
        super().__init__(num_classes, self.FEATURE_DIM)
        self.base = resnet18(pretrained=True)


class Resnet50FaceModel(ResnetFaceModel):

    FEATURE_DIM = 2048

    def __init__(self, num_classes):
        super().__init__(num_classes, self.FEATURE_DIM)
        self.base = resnet50(pretrained=True)