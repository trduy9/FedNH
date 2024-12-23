from torch import nn
import torch
from ..utils import autoassign, calculate_model_size, calculate_flops
from tqdm import tqdm, trange
from .utils import setup_optimizer


class Model(nn.Module):
    """For classification problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, dataloader):
        raise NotImplementedError

    def set_params(self, model_state_dict, exclude_keys=set()):
        """
            Reference: Be careful with the state_dict[key].
            https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
        """
        with torch.no_grad():
            for key in model_state_dict.keys():
                if key not in exclude_keys:
                    self.state_dict()[key].copy_(model_state_dict[key])


class ModelWrapper(Model):
    def __init__(self, base, head, config):
        """
            head and base should be nn.module
        """
        super(ModelWrapper, self).__init__(config)

        self.base = base
        self.head = head

    def forward(self, x, return_embedding):
        feature_embedding = self.base(x)
        out = self.head(feature_embedding)
        if return_embedding:
            return feature_embedding, out
        else:
            return out


class PhuongModelNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']

        self.conv1 = nn.Conv2d(3,32,kernel_size=4,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=4,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128,128,kernel_size=4,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2= nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fc1 = nn.Linear(6*6*128,512)
        self.fc2 = nn.Linear(512,config['num_classes'])
        
        temp = nn.Linear(512, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)

        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
        
        
        
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
        x = self.fc2(x)
        logits = self.prototype(x)
        return x, logits