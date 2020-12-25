# PFL-MoE: Personalized Federated Learning Based on Mixture of Experts

In our experiments, we use two image recognition datasets to conduct model training: 
Fashion-MNIST and CIFAR-10. With two network models trained, we have three combinations: Fashion-MNIST + LeNet-5, CIFAR-10 + LeNet-5, and CIFAR-10 + VGG-16. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run
$\alpha=[0.5, 0.9, 2.0]$ for each group of dataset+model
Stand-alone training experiments:
> python [main_local.py](main_local.py) --dataset fmnist --model lenet --epochs 100 --gpu 0 --num_users 100 --alpha 0.5 

> python [main_local.py](main_local.py) --dataset cifar --model lenet --epochs 100 --gpu 0 --num_users 100 --alpha 0.9

> python [main_local.py](main_local.py) --dataset cifar --model vgg --epochs 100 --gpu 0 --num_users 100 --alpha 2.0
 
FedAvg:
> python [main_fed.py](main_fed.py) --dataset fmnist --model lenet --epochs 1000 --gpu 0 --lr 0.01 --num_users 100 --frac 0.1 --alpha 0.5

> python [main_fed.py](main_fed.py) --dataset cifar --model lenet --epochs 1000 --gpu 0 --lr 0.01 --num_users 100 --frac 0.1 --alpha 0.9

> python [main_fed.py](main_fed.py) --dataset cifar --model vgg --epochs 1000 --gpu 0 --lr 0.01 --num_users 100 --frac 0.1 --alpha 2.0

PFL-FB:
> python [main_per_fb.py](main_per_fb.py) --dataset fmnist --model lenet --epochs 200 --gpu 0 --num_users 100 --alpha 0.5

> python [main_per_fb.py](main_per_fb.py) --dataset cifar --model lenet --epochs 200 --gpu 0 --num_users 100 --alpha 0.9

> python [main_per_fb.py](main_per_fb.py) --dataset cifar --model vgg --epochs 200 --gpu 0 --num_users 100 --alpha 2.0

PFL-MF:
> python [main_gate.py](main_gate.py) --dataset fmnist --model lenet --epochs 200 --num_users 100 --gpu 1 --alpha 0.5

> python [main_gate.py](main_gate.py) --dataset cifar --model lenet --epochs 200 --num_users 100 --gpu 1 --alpha 0.9

> python [main_gate.py](main_gate.py) --dataset cifar --model vgg --epochs 200 --num_users 100 --gpu 1 --alpha 2.0

PFL-MFE:
> python [main_gate.py](main_gate.py) --dataset fmnist --model lenet --epochs 200 --num_users 100 --gpu 1 --alpha 0.5 --struct

> python [main_gate.py](main_gate.py) --dataset cifar --model lenet --epochs 200 --num_users 100 --gpu 1 --alpha 0.9 --struct

> python [main_gate.py](main_gate.py) --dataset cifar --model vgg --epochs 200 --num_users 100 --gpu 1 --alpha 2.0 --struct

See the arguments in [options.py](utils/options.py). 
## Results
### 
Each client has two types of tests, including local test and global test. 

Table 1. The average value of **local test** accuracy of all clients in three baselines and proposed algorithms. Bold means the best in all methods.

|                              | non-IID $\alpha$ | Stand-alone <br />Traing(%) | FedAvg(%) | PFL-FB(%) | PFL-MF(%) | PFL-MFE(%) |
| :--------------------------: | :--------------: | :-------------------------: | :-------: | :-------: | --------: | :--------: |
|                              |       0.5        |            84.87            |    90     |   92.84   |     92.85 | **92.89**  |
| Fashion-MNIST & <br />LeNet5 |       0.9        |            82.23            |   90.31   |   91.84   | **92.02** |   92.01    |
|                              |        2         |            78.63            |   90.5    |   90.47   | **90.97** |   90.93    |
|                              |       0.5        |            65.58            |   68.92   | **77.46** |     75.49 |   77.23    |
|   CIFAR-10 & <br />LeNet5    |       0.9        |            61.49            |   70.7    |   74.7    |      74.1 | **74.74**  |
|                              |        2         |            55.8             |   72.69   |   72.5    |     73.24 | **73.44**  |
|                              |       0.5        |            52.77            |   88.16   | **91.92** |     90.63 |   91.71    |
|   CIFAR-10 &<br /> VGG-16    |       0.9        |            45.24            |   88.45   | **91.34** |     90.63 |   91.18    |
|                              |        2         |            34.2             |   89.17   | **90.4**  |     90.15 |    90.4    |

Table 2. The average value of **global test** accuracy of all clients. Bold means the best in all personalization algorithms.

|                              | non-IID $\alpha$ | Stand-alone <br />Traing(%) | FedAvg(%) | PFL-FB(%) | PFL-MF(%) | PFL-MFE(%) |
| :--------------------------: | :--------------: | :-------------------------: | :-------: | :-------: | :-------: | :--------: |
|                              |       0.5        |            57.77            |    90     |   83.35   | **85.45** |    85.3    |
| Fashion-MNIST & <br />LeNet5 |       0.9        |            65.28            |   90.31   |   85.91   | **87.69** |   87.67    |
|                              |        2         |            71.06            |   90.5    |   87.77   | **89.37** |   89.18    |
|                              |       0.5        |            28.89            |   68.92   |   54.28   | **62.33** |   58.27    |
|   CIFAR-10 &<br /> LeNet5    |       0.9        |            32.1             |   70.7    |   59.93   | **65.78** |   64.13    |
|                              |        2         |            35.32            |   72.69   |   66.06   | **69.79** |   69.78    |
|                              |       0.5        |            21.53            |   88.16   |   82.39   | **85.81** |   84.05    |
|   CIFAR-10 &<br /> VGG-16    |       0.9        |            22.45            |   88.45   |   82.62   | **88.15** |    87.9    |
|                              |        2         |            21.27            |   89.17   |   88.77   | **89.3**  |  **89.3**  |


## Acknowledgements

The code developed in this repo was was adapted from https://github.com/shaoxiongji/federated-learning.
