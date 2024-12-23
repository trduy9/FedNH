# FedNH

The application of federated learning models in supporting the prediction and classification of brain tumor diseases while ensuring data privacy and security, as well as addressing the impact of non-IID data and class imbalance in federated learning models.

## Run code

```
python main.py  --purpose Cifar --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
```

```
python main.py  --purpose Cifar --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
```

```
python main.py  --purpose Cifar --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/Cifar100_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
```

```
python main.py  --purpose Cifar --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/Cifar100_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
```

```
python main.py  --purpose Cifar --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/tumorMRI_PhuongModel.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
```

```
python main.py  --purpose Cifar --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/tumorMRI_PhuongModel.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
```

## Prepare Dataset

Please create a folder `data` under the root directory.

```
mkdir ~/data
```

* Cifar10, Cifar100: No extra steps are required.

* TinyImageNet
 * Download the dataset `cd ~/data && wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`
 * Unzip the file `unzip tiny-imagenet-200.zip`

## Run scripts
We prepared a python file `/experiments/gen_script.py` to generate bash commands to run experiments.

To reproduce the results for Cifar10/Cifar100, just set the variable `purpose` to `Cifar` in the `gen_script.py` file. Similarly, set `purpose` to `TinyImageNet` to run experiments for TinyImageNet.

`gen_script.py` will create a set of bash files named as `[method]_dir.sh`. Then use, for example, `bash FedAvg.sh` to run experiments.

We include a set of bash files to run experiments on `Cifar` in this submission.

## Organization of the code
The core code can be found at `src/flbase/`. Our framework builds upon three abstrac classes `server`, `clients`, and `model`. And their concrete implementations can be found in `models` directory and the `startegies` directory, respectively.

* `src/flbase/models`: We implemented or borrowed the implementation of (1) Convolution Neural Network and (2) Resnet18.
* `src/flbase/strategies`: We implement `CReFF`, `Ditto`, `FedAvg`, `FedBABU`, `FedNH`, `FedPer`, `FedProto`, `FedRep`, `FedROD`. Each file provides the concrete implementation of the corresponding `server` class and `client` class.

Helper functions, for example, generating non-iid data partition, can be found in `src/utils.py`.

-----

## Model CNN

Model using in testbed

1. Model Conv2 Layer

- Use normal

    ```jsx
    # Conv2Layer
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
    ```
- With prototypes

    ```jsx
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
    ```

------