import requests
import json
from specialization.evaluation import math_evaluator

url = "http://127.0.0.1:8000"
zero_shot_prompt = "要求：请你一步一步仔细思考并解答下列问题。\n问题是："
write_file = "glm2_math_test_result_pure.json"

with open('data/math23k_test.json', 'r', encoding='utf8')as fp:
    lines = fp.readlines()
n = len(lines)
i = int(0)
correct_n = int(0)
total_n = int(0)
while i < n:
    s = ''
    for j in range(7):
        s += lines[i+j]
    json_data = json.loads(s)
    q = json_data['original_text']
    eid = json_data['id']
    # q += "Let's think step by step."
    cor_ans = json_data['ans']
    # print("line", int(i/7), ":", q)
    # prompt = zero_shot_prompt + q
    prompt = q
    d = {"prompt": prompt, "history": [], "temperature": 0.97}
    r = requests.post(url, data=json.dumps(d))
    # print(r.text)
    res_content = json.loads(r.text)['response']
    glm_ans = math_evaluator.get_ans(res_content)
    point = math_evaluator.check_ans(glm_ans, cor_ans)
    correct_n += point
    total_n += 1
    print("id:", eid, "glm_answer:", glm_ans, " correct_answer:", cor_ans,
          " point:", point, " total_accuracy:", correct_n, "/", total_n, "=", correct_n / total_n)
    write_dict = {'id': eid, 'prompt': q, 'content': res_content, 'glm_answer': glm_ans,
                  'correct_answer': cor_ans}
    write_line = json.dumps(write_dict, ensure_ascii=False)
    with open(write_file, 'a', encoding='utf-8') as rf:
        rf.write(write_line+'\n')
    i += 7
