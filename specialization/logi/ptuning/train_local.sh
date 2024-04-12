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

