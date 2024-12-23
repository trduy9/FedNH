from ..model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from .ResNet import ResNet18, ResNet18NoNorm


class Conv2Cifar(Model):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 53 * 53, 384)
        self.linear2 = nn.Linear(384, 192)
        # intentionally remove the bias term for the last linear layer for fair comparison
        self.prototype = nn.Linear(192, config['num_classes'], bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 53 * 53)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 53 * 53)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return x, logits


class Conv2CifarNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 53 * 53, 384)
        self.linear2 = nn.Linear(384, 192)
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 53 * 53)
        x = F.relu(self.linear1(x))
        feature_embedding = F.relu(self.linear2(x))
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


# class ResNetMod(Model):
#     def __init__(self, config):
#         super().__init__(config)
#         if config['no_norm']:
#             self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
#         else:
#             self.backbone = ResNet18(num_classes=config['num_classes'])
#         self.prototype = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False)
#         self.backbone.linear = None

#     def forward(self, x):
#         # Convolution layers
#         feature_embedding = self.backbone(x)
#         logits = self.prototype(feature_embedding)
#         return logits

#     def get_embedding(self, x):
#         feature_embedding = self.backbone(x)
#         logits = self.prototype(feature_embedding)
#         return feature_embedding, logits


# class ResNetModNH(Model):
#     def __init__(self, config):
#         super().__init__(config)
#         self.return_embedding = config['FedNH_return_embedding']
#         if config['no_norm']:
#             self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
#         else:
#             self.backbone = ResNet18(num_classes=config['num_classes'])
#         temp = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False).state_dict()['weight']
#         self.prototype = nn.Parameter(temp)
#         self.backbone.linear = None
#         self.scaling = torch.nn.Parameter(torch.tensor([20.0]))
#         self.activation = None

#     def forward(self, x):
#         feature_embedding = self.backbone(x)
#         feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
#         if self.prototype.requires_grad == False:
#             normalized_prototype = self.prototype
#         else:
#             prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#             normalized_prototype = torch.div(self.prototype, prototype_norm)
#         logits = torch.matmul(feature_embedding, normalized_prototype.T)
#         logits = self.scaling * logits
#         self.activation = self.backbone.activation
#         if self.return_embedding:
#             return feature_embedding, logits
#         else:
#             return logits

class tumorModel_runCifarDataset_FedNH(Model):
    def __init__(self, config):
        super().__init__(config)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.prototype = nn.Linear(512, config['num_classes'])
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return x, logits


class tumorModel(Model):
    def __init__(self,config):
        super().__init__(config)
        
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=0) # kernel_siae 4->3
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(6*6*128,512)
        self.prototype = nn.Linear(512,config['num_classes'])
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
        
        
        
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return x, logits


class tumorModelNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.linear1 = nn.Linear(6*6*128, 512)
        self.linear2 = nn.Linear(512, 192)
        
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # print(f"Input size: {x.size()}")
        x = self.relu(self.bn1(self.conv1(x)))
        # print(f"After Conv1 and BN1: {x.size()}")
        x = self.pool(x)
        # print(f"After Pool1: {x.size()}")
        x = self.relu(self.bn2(self.conv2(x)))
        # print(f"After Conv2 and BN2: {x.size()}")
        x = self.pool(x)
        # print(f"After Pool2: {x.size()}")
        x = self.relu(self.bn3(self.conv3(x)))
        # print(f"After Conv3 and BN3: {x.size()}")
        x = self.pool2(x)
        # print(f"After Pool2_2: {x.size()}")
        x = self.relu(self.bn4(self.conv4(x)))
        # print(f"After Conv4 and BN4: {x.size()}")
        x = self.flatten(x)
        # print(f"After Flatten: {x.size()}")
        x = self.relu(self.linear1(x))
        # print(f"After FC1: {x.size()}")
        feature_embedding = self.dropout(x)
        feature_embedding = F.relu(self.linear2(feature_embedding))
        # print(f"After Linear2: {feature_embedding.size()}")
        
        # Normalize feature embedding
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        # print(f"After Normalizing feature embedding: {feature_embedding.size()}")

        # Normalize prototype
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        # print(f"Normalized Prototype size: {normalized_prototype.size()}")

        # Calculate logits
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits
        # print(f"Logits size: {logits.size()}")
        
        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


# class tumorModel_cifar10_dataNH(Model):
#     def __init__(self, config):
#         super().__init__(config)
#         self.return_embedding = config['FedNH_return_embedding']
        
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
        
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
        
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
        
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
#         self.linear1 = nn.Linear(6*6*128, 512)
#         self.linear2 = nn.Linear(512, 192)
        
#         temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
#         self.prototype = nn.Parameter(temp)
#         self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
#         self.flatten = nn.Flatten()
#         self.relu = nn.ReLU() 
#         self.dropout = nn.Dropout(0.5)
        
#     def forward(self, x):
#         print(x.size())
#         x = self.relu(self.bn1(self.conv1(x)))
#         print(x.size())
#         x = self.pool(x)
#         print(x.size())
#         x = self.relu(self.bn2(self.conv2(x)))
#         print(x.size())
#         x = self.pool(x)
#         print(x.size())
#         x = self.relu(self.bn3(self.conv3(x)))
#         print(x.size())
#         x = self.pool2(x)
#         print(x.size())
#         x = self.relu(self.bn4(self.conv4(x)))
#         print(x.size())
#         x = self.flatten(x)
#         print(x.size())
#         x = self.relu(self.linear1(x))
#         print(x.size())
#         feature_embedding = self.dropout(x)
#         print(x.size())
#         feature_embedding = F.relu(self.linear2(feature_embedding))
#         print(feature_embedding.size())
        
#         # Normalize feature embedding
#         feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         feature_embedding = torch.div(feature_embedding, feature_embedding_norm)

#         # Normalize prototype
#         if not self.prototype.requires_grad:
#             normalized_prototype = self.prototype
#         else:
#             prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#             normalized_prototype = torch.div(self.prototype, prototype_norm)

#         # Calculate logits
#         logits = torch.matmul(feature_embedding, normalized_prototype.T)
#         logits = self.scaling * logits
        
#         if self.return_embedding:
#             return feature_embedding, logits
#         else:
#             return logits

class tumorModel_cifar10_dataNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Corrected input size for self.linear1
        self.linear1 = nn.Linear(3 * 3 * 128, 512)
        self.linear2 = nn.Linear(512, 192)
        
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # print(x.size())
        x = self.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = self.relu(self.bn2(self.conv2(x)))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = self.relu(self.bn3(self.conv3(x)))
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())
        x = self.relu(self.bn4(self.conv4(x)))
        # print(x.size())
        x = self.flatten(x)
        # print(x.size())
        x = self.relu(self.linear1(x))
        # print(x.size())
        feature_embedding = self.dropout(x)
        # print(x.size())
        feature_embedding = F.relu(self.linear2(feature_embedding))
        # print(feature_embedding.size())
        
        # Normalize feature embedding
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)

        # Normalize prototype
        if not self.prototype.requires_grad:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)

        # Calculate logits
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits
        
        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits

class tumorModel_cifar10_dataProto(Model):
    def __init__(self,config):
        super().__init__(config)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(4*4*128, 512)  # Adjusted for CIFAR-10 image size
        self.prototype = nn.Linear(512, config['num_classes'], bias=False)
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
    def get_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return x, logits
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return logits