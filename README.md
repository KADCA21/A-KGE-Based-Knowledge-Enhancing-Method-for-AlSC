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
-run.sh or python ./main_total.py

## Notes

-Half of our code are based on DM-GCN(Pang et.al, ACL2021) and KGAN(Zhong et.al, arXiv preprint)

