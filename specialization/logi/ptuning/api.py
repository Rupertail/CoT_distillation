import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_model(version: str, checkpoint: int):
    """load fine-tuned model, version: test, normal, classified. RETURN: tokenizer, model"""
    original_path = os.path.join('E:\\LLM\\ChatGLM\\chatglm2-6b-int4\\ChatGLM2-6B-main\\specialization\\logi\\output')
    if version == 'test':
        type_path = os.path.join(original_path, 'logi-test-pt-32-2e-2')
    elif version == 'normal':
        type_path = os.path.join(original_path, 'logi-normal-pt-1024-1e-3')
    elif version == 'classified':
        type_path = os.path.join(original_path, 'logi-classified-pt-32-2e-2')
    else:
        return None
    CHECKPOINT_PATH = os.path.join(type_path, 'checkpoint-'+str(checkpoint))
    # 载入Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("E:\\LLM\\ChatGLM\\chatglm2-6b-int4\\ChatGLM2-6B-main\\model", trust_remote_code=True)
    config = AutoConfig.from_pretrained("E:\\LLM\\ChatGLM\\chatglm2-6b-int4\\ChatGLM2-6B-main\\model", trust_remote_code=True, pre_seq_len=1024)
    model = AutoModel.from_pretrained("E:\\LLM\\ChatGLM\\chatglm2-6b-int4\\ChatGLM2-6B-main\\model", config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.quantize(4)
    model = model.cuda()
    model = model.eval()
    return tokenizer, model
    # api use example:
    # response, history = model.chat(tokenizer, "你好", history=[])
