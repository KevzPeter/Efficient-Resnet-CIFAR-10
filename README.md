# CIFAR 10 Classification using Efficient Resnet Architecture

An efficient re-implementation of Resnet-18 Architecture that produces higher test accuracy with under 5 million parameters.

Team Name: [object Object]
Team Members: Kevin Peter (kevin.peter@nyu.edu)
NetID: kpk4354
Kaggle Competition: [https://www.kaggle.com/competitions/deep-learning-spring-2025-project-1](https://www.kaggle.com/competitions/deep-learning-spring-2025-project-1)

[Click here to view the notebook](./Deep_Learning_Spring_2025_CIFAR_10_classification.ipynb)

### Model summary

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4           [-1, 64, 32, 32]          36,928
       BatchNorm2d-5           [-1, 64, 32, 32]             128
         Dropout2d-6           [-1, 64, 32, 32]               0
            Conv2d-7           [-1, 64, 32, 32]          36,928
       BatchNorm2d-8           [-1, 64, 32, 32]             128
ResidualBlockWithDropout-9           [-1, 64, 32, 32]               0
           Conv2d-10           [-1, 64, 32, 32]          36,928
      BatchNorm2d-11           [-1, 64, 32, 32]             128
        Dropout2d-12           [-1, 64, 32, 32]               0
           Conv2d-13           [-1, 64, 32, 32]          36,928
      BatchNorm2d-14           [-1, 64, 32, 32]             128
ResidualBlockWithDropout-15           [-1, 64, 32, 32]               0
           Conv2d-16           [-1, 64, 32, 32]          36,928
      BatchNorm2d-17           [-1, 64, 32, 32]             128
        Dropout2d-18           [-1, 64, 32, 32]               0
           Conv2d-19           [-1, 64, 32, 32]          36,928
      BatchNorm2d-20           [-1, 64, 32, 32]             128
ResidualBlockWithDropout-21           [-1, 64, 32, 32]               0
           Conv2d-22           [-1, 64, 32, 32]          36,928
      BatchNorm2d-23           [-1, 64, 32, 32]             128
        Dropout2d-24           [-1, 64, 32, 32]               0
           Conv2d-25           [-1, 64, 32, 32]          36,928
      BatchNorm2d-26           [-1, 64, 32, 32]             128
ResidualBlockWithDropout-27           [-1, 64, 32, 32]               0
           Conv2d-28          [-1, 128, 16, 16]           8,320
      BatchNorm2d-29          [-1, 128, 16, 16]             256
           Conv2d-30          [-1, 128, 16, 16]          73,856
      BatchNorm2d-31          [-1, 128, 16, 16]             256
        Dropout2d-32          [-1, 128, 16, 16]               0
           Conv2d-33          [-1, 128, 16, 16]         147,584
      BatchNorm2d-34          [-1, 128, 16, 16]             256
ResidualBlockWithDropout-35          [-1, 128, 16, 16]               0
           Conv2d-36          [-1, 128, 16, 16]         147,584
      BatchNorm2d-37          [-1, 128, 16, 16]             256
        Dropout2d-38          [-1, 128, 16, 16]               0
           Conv2d-39          [-1, 128, 16, 16]         147,584
      BatchNorm2d-40          [-1, 128, 16, 16]             256
ResidualBlockWithDropout-41          [-1, 128, 16, 16]               0
           Conv2d-42          [-1, 128, 16, 16]         147,584
      BatchNorm2d-43          [-1, 128, 16, 16]             256
        Dropout2d-44          [-1, 128, 16, 16]               0
           Conv2d-45          [-1, 128, 16, 16]         147,584
      BatchNorm2d-46          [-1, 128, 16, 16]             256
ResidualBlockWithDropout-47          [-1, 128, 16, 16]               0
           Conv2d-48          [-1, 128, 16, 16]         147,584
      BatchNorm2d-49          [-1, 128, 16, 16]             256
        Dropout2d-50          [-1, 128, 16, 16]               0
           Conv2d-51          [-1, 128, 16, 16]         147,584
      BatchNorm2d-52          [-1, 128, 16, 16]             256
ResidualBlockWithDropout-53          [-1, 128, 16, 16]               0
           Conv2d-54            [-1, 256, 8, 8]          33,024
      BatchNorm2d-55            [-1, 256, 8, 8]             512
           Conv2d-56            [-1, 256, 8, 8]         295,168
      BatchNorm2d-57            [-1, 256, 8, 8]             512
        Dropout2d-58            [-1, 256, 8, 8]               0
           Conv2d-59            [-1, 256, 8, 8]         590,080
      BatchNorm2d-60            [-1, 256, 8, 8]             512
ResidualBlockWithDropout-61            [-1, 256, 8, 8]               0
           Conv2d-62            [-1, 256, 8, 8]         590,080
      BatchNorm2d-63            [-1, 256, 8, 8]             512
        Dropout2d-64            [-1, 256, 8, 8]               0
           Conv2d-65            [-1, 256, 8, 8]         590,080
      BatchNorm2d-66            [-1, 256, 8, 8]             512
ResidualBlockWithDropout-67            [-1, 256, 8, 8]               0
           Conv2d-68            [-1, 256, 8, 8]         590,080
      BatchNorm2d-69            [-1, 256, 8, 8]             512
        Dropout2d-70            [-1, 256, 8, 8]               0
           Conv2d-71            [-1, 256, 8, 8]         590,080
      BatchNorm2d-72            [-1, 256, 8, 8]             512
ResidualBlockWithDropout-73            [-1, 256, 8, 8]               0
AdaptiveAvgPool2d-74            [-1, 256, 1, 1]               0
           Linear-75                   [-1, 10]           2,570
================================================================
Total params: 4,700,682
Trainable params: 4,700,682
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 22.50
Params size (MB): 17.93
Estimated Total Size (MB): 40.45
----------------------------------------------------------------
```

### Final Hyperparameters

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
| **Total number of Parameters** | **4,700,682**     | **11,173,962**      |

Paper cited: [Efficient ResNets: Residual Network Design](https://arxiv.org/abs/2306.12100)
