# JDVAE 

Pytorch implementation of the paper [**JOINT DISENTANGLEMENT OF LABELS AND THEIR FEATURES WITH VAE (ICIP 2022)**](https://hal.science/hal-03780425/file/ICIP2022.pdf)


#### Bibtex
If you find this code useful in your research, please cite:

```
@inproceedings{zou2022joint,
  title={Joint disentanglement of labels and their features with VAE},
  author={Zou, Kaifeng and Faisan, Sylvain and Heitz, Fabrice and Valette, Sebastien},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1341--1345},
  year={2022},
  organization={IEEE}
}

```
### 1. Requirements
This code is test on Python3.7.11 and pytorch1.9+cu111. 

### 2. Download the datasets
We use CelebA faces attributes dataset, you can download it from [**Kaggle**](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and then move them to 'data/'.


### 3. Training

```bash
python main_vae.py
```

### 4. Testing
Note that the code need to train a classifier to calculate the successful rate of attribute swapping. 
```bash
python test.py
```


### 5. Results
#### Comparison

<img src='results/comp_exp.png' width='512'>

#### Multi-label manipulation
<img src='results/multi_exp.png' width='512'>

