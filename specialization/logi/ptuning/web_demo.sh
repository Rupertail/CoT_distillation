PRE_SEQ_LEN=32

CUDA_VISIBLE_DEVICES=0 python web_demo.py \
    --model_name_or_path E:\\LLM\\ChatGLM\\chatglm2-6b-int4\\ChatGLM2-6B-main\\model \
    --ptuning_checkpoint ../output/logi-test-pt-32-2e-2/checkpoint-3000 \
    --pre_seq_len $PRE_SEQ_LEN

