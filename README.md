# A-KGE-Based-Knowledge-Enhancing-Method-for-AlSC
## Environment
-OS Ubuntu 16.04.6 LTS  
-python 3.7.6  
-GPU: NVIDIA GeForce RTX 3090(Driver version:460.56, CUDA Version:11.2)  

## Dataset
-Semeval14 laptop  
-Semeval14 rest  
-Twitter  

## Parameters
-ds_name: name of the dataset, options:14semeval_rest 14semeval_laptop Twitter  
-bs: batch_size  
-learning_rate: default 0.001  
-n_epoch: number of epoch for training, default 20  
-test: whether test model or not, default 0  

## Usage
### Traning the model:
-run.sh or python ./main.py

## Notes

-Half of our code are based on DM-GCN(Pang et.al, ACL2021) and KGAN(Zhong et.al, arXiv preprint)

## Citation
```
@Article{math10203908,
AUTHOR = {Yu, Haibo and Lu, Guojun and Cai, Qianhua and Xue, Yun},
TITLE = {A KGE Based Knowledge Enhancing Method for Aspect-Level Sentiment Classification},
JOURNAL = {Mathematics},
VOLUME = {10},
YEAR = {2022},
NUMBER = {20},
ARTICLE-NUMBER = {3908},
URL = {https://www.mdpi.com/2227-7390/10/20/3908},
ISSN = {2227-7390},
DOI = {10.3390/math10203908}
}
```
