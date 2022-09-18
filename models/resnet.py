import torch
from torch import nn
from torchvision.models import resnet18, resnet50

from .base import FaceModel
from device import device


class ResnetFaceModel(FaceModel):

    IMAGE_SHAPE = (112, 112)

    def __init__(self, num_classes, feat_normalization, feature_dim):
        super().__init__(num_classes, feature_dim)

        self.feat_normalization = feat_normalization
        self.avgpool = nn.AvgPool2d((4, 4))
        self.bn2 = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.bn3 = nn.BatchNorm1d(512)

        self.extract_feature = nn.Linear(
            2048, self.feature_dim)
        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.bn2(x) # optional
        x = self.dropout(x) # optional

        batch_size = x.size(0)
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        feature = self.extract_feature(x)
        if self.feat_normalization == 'batchnorm':
            feature = self.bn3(feature)
        logits = self.classifier(feature) if self.num_classes else None

        if self.feat_normalization == 'l2':
            feature_normed = feature.div(
                torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

            return logits, feature_normed
        else:
            return logits, feature


class Resnet18FaceModel(ResnetFaceModel):

    FEATURE_DIM = 512

    def __init__(self, num_classes):
        super().__init__(num_classes, self.FEATURE_DIM)
        self.base = resnet18(pretrained=True)


class Resnet50FaceModel(ResnetFaceModel):

    FEATURE_DIM = 512


    def __init__(self, num_classes, feat_normalization):
        super().__init__(num_classes, feat_normalization, self.FEATURE_DIM)
        self.backbone = resnet50(pretrained=True)
        #self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # this keeps adaptive pooling layer
        #self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # this removes adaptive pooling layer