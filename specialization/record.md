##### teacher

| model       | dataset       | prompt         | T    | accuracy |
| ----------- | ------------- | -------------- | ---- | -------- |
| glm-3-turbo | math23k-test  | zs-CoT-1       | 0.97 | 0.782    |
| glm-3-turbo | math23k-test  | zs-CoT-2       | 0.97 | 0.823    |
| glm-3-turbo | LogiQA-test   | zs-CoT         | 0.95 | 0.666    |
| glm-4       | math23k-test  | zs-CoT-1       | 0.97 | 0.796    |
| glm-4       | math23k-test  | zs-CoT-2       | 0.97 | 0.906    |
| glm-4       | math23k-test  | fs-CoT         | 0.97 | 0.885    |
| glm-4       | math23k-train | zs-CoT-2       | 0.97 | 0.91156  |
| glm-4       | math23k-train | classified-CoT | 0.5  | 0.8385   |
| glm-4       | LogiQA-test   | zs-CoT         | 0.95 | 0.738    |
| glm-4       | LogiQA-test   | zs-CoT         | 0.97 | 0.737    |
| glm-4       | LogiQA-train  | zs-CoT         | 0.95 | 0.7365   |

##### student (Chatglm2-6B)

| version  | dataset      | prompt | T    | accuracy |
| -------- | ------------ | ------ | ---- | -------- |
| original | math23k-test | zs-CoT | 0.97 | 0.461    |
| original | math23k-test | (None) | 0.97 | 0.465    |
| original | math23k-test | fs-CoT | 0.97 | 0.386    |
| original | LogiQA-test  | zs-CoT | 0.95 | 0.4103   |
| original | LogiQA-test  | (None) | 0.95 | 0.4247   |

##### logi-normal-pt-32-2e-2

![image-20240406102403485](C:\Users\王俊哲\AppData\Roaming\Typora\typora-user-images\image-20240406102403485.png)

![image-20240407124139282](C:\Users\王俊哲\AppData\Roaming\Typora\typora-user-images\image-20240407124139282.png)

```Python
'''
{
 "prompt": 
 "甲乙两车相距20千米，乙车在前，甲车在后，两车同时出发，2小时后甲车追上乙车，已知乙车每小时行50千米，甲车每小时行多少千米．",
 "content": "
 【问题分类】
 追及问题
 【含义】
 两个运动物体在不同地点同时出发（或者在同一地点而不是同时出发，或者在不同地点又不是同时出发）作同向运动，在后面的，行进速度要快些，在前面的，行进速度较慢些，在一定时间之内，后面的追上前面的物体。这类应用题就叫做追及问题。
 【数量关系】
 追及时间＝追及路程÷（快速－慢速）\n追及路程＝（快速－慢速）×追及时间
 【解题思路和方法】
 简单的题目直接利用公式，复杂的题目变通后利用公式。
 【解题步骤】
 甲乙两车相距20千米，2小时后甲车追上乙车，乙车每小时行50千米。\n\n首先，根据追及问题的数量关系，追及时间＝追及路程÷（快速－慢速），这里的追及路程为20千米，追及时间为2小时。\n\n然后，将已知数值代入公式计算甲车的速度：\n20千米 = (甲车速度 - 乙车速度) × 2小时\n\n接下来，解这个方程求甲车速度：\n甲车速度 - 乙车速度 = 20千米 / 2小时\n甲车速度 - 50千米/小时 = 10千米/小时\n甲车速度 = 10千米/小时 + 50千米/小时\n甲车速度 = 60千米/小时
 【答案】
 60
 "}
'''
```

```shell
PRE_SEQ_LEN=512
LR=1e-2
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --train_file ../data/normal_train.json \
    --validation_file ../data/normal_train.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column content \
    --overwrite_cache \
    --model_name_or_path E:\\LLM\\ChatGLM\\chatglm2-6b-int4\\ChatGLM2-6B-main\\model \
    --output_dir ../output/logi-normal-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 6000 \
    --logging_steps 10 \
    --save_steps 300 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4


```

