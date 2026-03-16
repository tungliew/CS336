# Stanford CS336 Assignment5

## Data Source
数据集来自Kaggle MATH
train = 7.5k, test = 5k 
没有sft数据
可自行下载

## Supervised Finetuning
sft_microbatch_train_step（）没有通过测试
其他helper methods通过测试
![alt text](/Users/tungliew/Desktop/sft.png)

## Group Relative Policy Optimization
全部通过测试
![alt text](/Users/tungliew/Desktop/grpo.png)

## grpo_train.py
GPU太小影响训练，可能调小sampling_max_tokens可以内存适应，但是token太小又会影响reward判断
后续不会进行各种消融实验
