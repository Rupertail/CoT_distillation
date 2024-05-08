import os
import sys
# sys.path.append("/gpfsdata/home/wangjunzhe/CoT_distillation")
import requests
import json
import specialization.evaluation.math_evaluator as me
import time

# original url:
url = "http://127.0.0.1:8000/v1/chat/completions"
zero_shot_prompt = "要求：请你一步一步仔细思考并解答下列问题。\n问题是："

with open('data/math23k_test.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
i = int(0)
write_file = "glm2_math_test_result_lora_normal_5e-4-6500.json"

start_time = time.time()

correct_n = int(0)
total_n = int(0)
while i < n:
    s = ''
    for j in range(7):
        s += lines[i + j]
    json_data = json.loads(s)
    q = json_data['original_text']
    eid = json_data['id']
    cor_ans = json_data['ans']
    # zero_shot_CoT-prompt
    # p = zero_shot_prompt + q
    # None-prompt
    p = q
    messages = [{"role": "user", "content": p}]
    d = {"model": "chatglm", "messages": messages}
    r = requests.post(url, data=json.dumps(d))
    # print("r:", r.text)
    res_content = json.loads(r.text)["choices"][0]["message"]["content"]
    glm_ans = me.get_ans(res_content)
    point = me.check_ans(glm_ans, cor_ans)
    correct_n += point
    total_n += 1
    print("id:", eid, "glm_answer:", glm_ans, " correct_answer:", cor_ans,
          " point:", point, " total_accuracy:", correct_n, "/", total_n, "=", correct_n / total_n)
    # print(r.text)
    write_dict = {'id': eid, 'prompt': p, 'content': res_content, 'glm_answer': glm_ans,
                  'correct_answer': cor_ans, 'point': point}
    write_line = json.dumps(write_dict, ensure_ascii=False)
    with open(write_file, 'a', encoding='utf-8') as rf:
        rf.write(write_line + '\n')
    i += 7

end_time = time.time()
used_time = end_time - start_time
print("totally used time: {.2f}s\n".format(used_time))
print("used time per instruction: {.2f}s\n".format(used_time/total_n))
print("Results saved in \'" + write_file + "\'.")
