# Stanford CS336 Assignment5

## Data Source
数据集来自Kaggle MATH<br>
train = 7.5k, test = 5k <br>
没有sft数据<br>
可自行下载<br>

## Supervised Finetuning
sft_microbatch_train_step（）没有通过测试<br>
其他helper methods通过测试<br>

## Group Relative Policy Optimization
全部通过测试<br>

## grpo_train.py
GPU太小影响训练，可能调小sampling_max_tokens可以内存适应，但是token太小又会影响reward判断<br>
后续不会进行各种消融实验<br>
