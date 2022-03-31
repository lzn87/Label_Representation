import models.cifar as models
import torch
import torch.nn as nn
import torchvision

class PytorchCategoryModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(PytorchCategoryModel, self).__init__()
        
        # CNN
        if model_name == 'vgg19':
            cnn = model = torchvision.models.vgg19_bn(pretrained=True)
            self.features = nn.Sequential(*list(cnn.children())[:-1],
                                         nn.Flatten(),
                                         nn.Linear(25088, 512))
            self.fc = nn.Linear(512, num_classes)
        elif model_name == 'resnet110':
            cnn = torchvision.models.resnet101(pretrained=True)
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Linear(cnn.fc.in_features, num_classes)
        elif model_name == 'resnet32':
            cnn = torchvision.models.resnet34(pretrained=True)
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Linear(cnn.fc.in_features, num_classes)
        else:
            print(f'model name {model_name} not recognized')

    def forward(self, data):
        x = self.features(data)
        x = x.view(x.size(0), -1)
        prediction = self.fc(x)
        return prediction, x
    
class PytorchHighDimensionalModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(PytorchHighDimensionalModel, self).__init__()
        
        # CNN
        if model_name == 'vgg19':
            cnn = model = torchvision.models.vgg19_bn()
            self.features = nn.Sequential(*list(cnn.children())[:-1], nn.Flatten(), nn.Linear(25088, 512))
            self.fc = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(512, 64))
        elif model_name == 'resnet110':
            cnn = torchvision.models.resnet101()
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.fc.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.fc.in_features, 64))
        elif model_name == 'resnet32':
            cnn = torchvision.models.resnet34()
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.fc.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.fc.in_features, 64))

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
    
    def forward(self, data):
        feat = self.features(data)
        feat = feat.view(feat.size(0), -1)
        x = self.fc(feat)
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        x = self.deconvs(x)
        prediction = x.view(-1, 64, 64)
        return prediction, feat

    
# A model predicts category labels.
class CategoryModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CategoryModel, self).__init__()
        
        # CNN
        if model_name == 'vgg19':
            cnn = models.__dict__['vgg19_bn'](num_classes=num_classes)
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Linear(cnn.classifier.in_features, num_classes)
        elif model_name == 'resnet110':
            cnn = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=110,
                block_name='BasicBlock')
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Linear(cnn.fc.in_features, num_classes)
        elif model_name == 'resnet32':
            cnn = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=32,
                block_name='BasicBlock')
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Linear(cnn.fc.in_features, num_classes)

    def forward(self, data):
        x = self.features(data)
        x = x.view(x.size(0), -1)
        prediction = self.fc(x)
        return prediction


# A model predicts high-dimensional labels.
class HighDimensionalModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(HighDimensionalModel, self).__init__()
        
        # CNN
        if model_name == 'vgg19':
            cnn = models.__dict__['vgg19_bn'](num_classes=num_classes)
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.classifier.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.classifier.in_features, 64))
        elif model_name == 'resnet110':
            cnn = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=110,
                block_name='BasicBlock')
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.fc.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.fc.in_features, 64))
        elif model_name == 'resnet32':
            cnn = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=32,
                block_name='BasicBlock')
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.fc.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.fc.in_features, 64))

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
    
    def forward(self, data):
        x = self.features(data)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        x = self.deconvs(x)
        prediction = x.view(-1, 64, 64)
        return prediction


# A model predicts high-dimensional labels.
class BERTHighDimensionalModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(BERTHighDimensionalModel, self).__init__()
        
        # CNN
        if model_name == 'vgg19':
            cnn = models.__dict__['vgg19_bn'](num_classes=num_classes)
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.classifier.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.classifier.in_features, 64))
        elif model_name == 'resnet110':
            cnn = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=110,
                block_name='BasicBlock')
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.fc.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.fc.in_features, 64))
        elif model_name == 'resnet32':
            cnn = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=32,
                block_name='BasicBlock')
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            self.fc = nn.Sequential(
                nn.BatchNorm1d(cnn.fc.in_features),
                nn.LeakyReLU(),
                nn.Linear(cnn.fc.in_features, 64))

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 3 x 3
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 6 x 6
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 12 x 12
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 24 x 24
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
        )
    
    def forward(self, data):
        x = self.features(data)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        x = self.deconvs(x)
        prediction = x.view(-1, 48, 48)
        return prediction
        