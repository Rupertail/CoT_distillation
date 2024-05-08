python src/api_demo.py \
    --model_name_or_path /gpfsdata/home/wangjunzhe/CoT_distillation/model/ \
    --finetuning_type lora \
    --checkpoint_dir /gpfsdata/home/wangjunzhe/CoT_distillation/specialization/math/output/math-normal-lora-5e-4/checkpoint-4000 \
    --quantization_bit 8