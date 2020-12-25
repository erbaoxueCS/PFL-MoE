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
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97.21%      | 93.03%         |
| FedAVG-CNN| 98.60%      | 93.81%         |


## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

Shaoxiong Ji, Shirui Pan, Guodong Long, Xue Li, Jing Jiang, and Zi Huang. Learning private neural language modeling with attentive aggregation. In the 2019 International Joint Conference on Neural Networks (IJCNN), 2019. [[Paper](https://arxiv.org/abs/1812.07108)] [[Code](https://github.com/shaoxiongji/fed-att)]

Jing Jiang, Shaoxiong Ji, and Guodong Long. Decentralized knowledge acquisition for mobile internet applications. World Wide Web, 2020. [[Paper](https://link.springer.com/article/10.1007/s11280-019-00775-w)]


