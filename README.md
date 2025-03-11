# CIFAR 10 Classification using Modified Resnet Architecture

### Hyperparameters

| **Parameter**                  | **My Model**      | **ResNet18**        |
| ------------------------------ | ----------------- | ------------------- |
| number of residual layers      | 3                 | 4                   |
| number of residual blocks      | [4, 4, 3]         | [2, 2, 2, 2]        |
| convolutional kernel sizes     | [3, 3, 3]         | [3, 3, 3, 3]        |
| shortcut kernel sizes          | [1, 1, 1]         | [1, 1, 1, 1]        |
| number of channels             | [64, 128, 256]    | [64, 128, 256, 512] |
| average pool kernel size       | 1                 | 4                   |
| batch normalization            | True              | True                |
| dropout                        | 0.4               | 0                   |
| data augmentation              | True              | False               |
| data normalization             | True              | False               |
| optimizer                      | SGD               | SGD                 |
| learning rate (lr)             | 0.1               | 0.1                 |
| lr scheduler                   | CosineAnnealingLR | CosineAnnealingLR   |
| weight decay                   | 0.0005            | 0.0005              |
| batch size                     | 128               | 128                 |
| number of workers              | 8                 | 16                  |
| **Total number of Parameters** | **4,697,742**     | **11,173,962**      |

Paper cited: [Efficient ResNets: Residual Network Design](https://arxiv.org/abs/2306.12100)
