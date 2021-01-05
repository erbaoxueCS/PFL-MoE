# PFL-MoE: Personalized Federated Learning Based on Mixture of Experts

In our experiments, we use two image recognition datasets to conduct model training: 
Fashion-MNIST and CIFAR-10. With two network models trained, we have three combinations: Fashion-MNIST + LeNet-5, CIFAR-10 + LeNet-5, and CIFAR-10 + VGG-16. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run
dataset+model: fmnist+lenet, cifar+lenet, cifar+vgg<br>
$\alpha=[0.5, 0.9, 2.0]$ for each group of dataset+model

Local:
> python [main_local.py](main_local.py) --dataset fmnist --model lenet --epochs 100 --gpu 0 --num_users 100 --alpha 0.5

FedAvg: 
> python [main_fed.py](main_fed.py) --dataset fmnist --model lenet --epochs 1000 --gpu 0 --lr 0.01 --num_users 100 --frac 0.1 --alpha 0.5

PFL-FB + PFL-MF:
> python [main_gate.py](main_gate.py) --dataset fmnist --model lenet --epochs 200 --num_users 100 --gpu 1 --alpha 0.5

PFL-FB + PFL-MFE:
> python [main_gate.py](main_gate.py) --dataset fmnist --model lenet --epochs 200 --num_users 100 --gpu 1 --alpha 0.5 --struct

See the arguments in [options.py](utils/options.py). 
## Results
### 
Each client has two types of tests, including local test and global test. 

Table 1. The average value of **local test** accuracy of all clients in three baselines and proposed algorithms. 

<table>
   <tr>
      <td></td>
      <td>non-IID</td>
      <td>Local(%)</td>
      <td>FedAvg(%)</td>
      <td>PFL-FB(%)</td>
      <td>PFL-MF(%)</td>
      <td>PFL-MFE(%)</td>
   </tr>
   <tr>
      <td rowspan="3">Fashion-MNIST & LeNet5</td><!--rowspan="3"纵向合并三个单元格-->
      <td>0.5</td>
      <td>84.87</td>
      <td>90</td>
      <td>92.84</td>
      <td>92.85</td>
      <td style="font-weight:bold">92.89</td>
   </tr>
   <tr>
      <td>0.9</td>
      <td>82.23</td>
      <td>90.31</td>
      <td>91.84</td>
      <td style="font-weight:bold">92.02</td>
      <td>92.01</td>
   </tr>
   <tr>
      <td>2</td>
      <td>78.63</td>
      <td>90.5</td>
      <td>90.47</td>
      <td style="font-weight:bold">90.97</td>
      <td>90.93</td>
   </tr>
   <tr>
      <td rowspan="3">CIFAR-10 & LeNet5</td>
      <td>0.5</td>
      <td>65.58</td>
      <td>68.92</td>
      <td style="font-weight:bold">77.46</td>
      <td>75.49</td>
      <td>77.23</td>
   </tr>
   <tr>
      <td>0.9</td>
      <td>61.49</td>
      <td>70.7</td>
      <td>74.7</td>
      <td>74.1</td>
      <td style="font-weight:bold">74.74</td>
   </tr>
   <tr>
      <td>2</td>
      <td>55.8</td>
      <td>72.69</td>
      <td>72.5</td>
      <td>73.24</td>
      <td style="font-weight:bold">73.44</td>
   </tr>
   <tr>
      <td rowspan="3">CIFAR-10 & VGG-16</td>
      <td>0.5</td>
      <td>52.77</td>
      <td>88.16</td>
      <td style="font-weight:bold">91.92</td>
      <td>90.63</td>
      <td>91.71</td>
   </tr>
   <tr>
      <td>0.9</td>
      <td>45.24</td>
      <td>88.45</td>
      <td style="font-weight:bold">91.34</td>
      <td>90.63</td>
      <td>91.18</td>
   </tr>
   <tr>
      <td>2</td>
      <td>34.2</td>
      <td>89.17</td>
      <td style="font-weight:bold">90.4</td>
      <td>90.15</td>
      <td style="font-weight:bold">90.4</td>
   </tr>
</table>

Table 2. The average value of **global test** accuracy of all clients. 

<table>
   <tr>
      <td></td>
      <td>non-IID </td>
      <td>Local(%)</td>
      <td>FedAvg(%)</td>
      <td>PFL-FB(%)</td>
      <td>PFL-MF(%)</td>
      <td>PFL-MFE(%)</td>
   </tr>
   <tr>
      <td rowspan="3">Fashion-MNIST & LeNet5</td>
      <td>0.5</td>
      <td>57.77</td>
      <td>90</td>
      <td>83.35</td>
      <td style="font-weight:bold">85.45</td>
      <td>85.3</td>
   </tr>
   <tr>
      <td>0.9</td>
      <td>65.28</td>
      <td>90.31</td>
      <td>85.91</td>
      <td style="font-weight:bold">87.69</td>
      <td>87.67</td>
   </tr>
   <tr>
      <td>2</td>
      <td>71.06</td>
      <td>90.5</td>
      <td>87.77</td>
      <td style="font-weight:bold">89.37</td>
      <td>89.18</td>
   </tr>
   <tr>
      <td rowspan="3">CIFAR-10 & LeNet5</td>
      <td>0.5</td>
      <td>28.89</td>
      <td>68.92</td>
      <td>54.28</td>
      <td style="font-weight:bold">62.33</td>
      <td>58.27</td>
   </tr>
   <tr>
      <td>0.9</td>
      <td>32.1</td>
      <td>70.7</td>
      <td>59.93</td>
      <td style="font-weight:bold">65.78</td>
      <td>64.13</td>
   </tr>
   <tr>
      <td>2</td>
      <td>35.32</td>
      <td>72.69</td>
      <td>66.06</td>
      <td style="font-weight:bold">69.79</td>
      <td>69.78</td>
   </tr>
   <tr>
      <td rowspan="3">CIFAR-10 & VGG-16</td>
      <td>0.5</td>
      <td>21.53</td>
      <td>88.16</td>
      <td>82.39</td>
      <td style="font-weight:bold">85.81</td>
      <td>84.05</td>
   </tr>
   <tr>
      <td>0.9</td>
      <td>22.45</td>
      <td>88.45</td>
      <td>82.62</td>
      <td style="font-weight:bold">88.15</td>
      <td>87.9</td>
   </tr>
   <tr>
      <td>2</td>
      <td>21.27</td>
      <td>89.17</td>
      <td>88.77</td>
      <td style="font-weight:bold">89.3</td>
      <td style="font-weight:bold">89.3</td>
   </tr>
</table>

Fig 1. Fashion-MNIST + LeNet-5, $\alpha=0.9$. The global test accuracy and local test accuracy of all client of PFL-FB, PFL-MF, and PFL-MFE algorithms. All x-axis are FedAvg local test accuracy of each client (can be regarded as client index). Each point represents a test accuracy comparison between a PFL algorithm and FedAvg for a particular client.

![fmnist_lenet_0.9](https://github.com/guobbin/PFL-MoE/blob/master/imgs/09fmnist_lenet.pdf)

Fig 2. CIFAR-10 + LeNet-5, $\alpha=0.9$.

![cifar_lenet_0.9](https://github.com/guobbin/PFL-MoE/blob/master/imgs/09cifar_lenet.pdf)

Fig 3. CIFAR-10 + VGG-16, $\alpha=2.0$.

![cifar_vgg_2.0](https://github.com/guobbin/PFL-MoE/blob/master/imgs/2cifar_vgg.pdf)

## Acknowledgements

Acknowledgments give to [shaoxiongji](https://github.com/shaoxiongji)