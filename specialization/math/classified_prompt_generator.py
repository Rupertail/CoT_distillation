import json

import zhipuai
from zhipuai import ZhipuAI
from specialization.evaluation import math_evaluator as me
import time
import config

f = open('../zhipu_api_key.txt')
api_key = f.readline().strip()  # 读取第一行
client = ZhipuAI(api_key=api_key)  # 填写您自己的APIKey
kid = "1773271276501307392"     # 知识库id
read_file = "data/math23k_train.json"
write_file = "classified_prompts.json"

with open(read_file, 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
group_n = int(50)  # question number in a group
i = int(39200)    # begin line
correct_n = int(0)
total_n = int(0)
group = []

while i <= n:
    if (i != 0) and ((i % (group_n * 7) == 0) or (i == n)):
        responses = []
        write_lines = []
        for k in range(len(group)):
            try:
                response = client.chat.asyncCompletions.create(
                    model="glm-4",  # 填写需要调用的模型名称
                    messages=[
                        {"role": "user", "content": group[k][0], "temperature": 0.5}
                    ],
                    tools=[
                        {
                            "type": "retrieval",
                            "retrieval": {
                                "knowledge_id": kid,
                                "prompt_template": config.retrieval_prompt()
                            }
                        }
                    ],
                )
                responses.append(response.id)
            except Exception:
                responses.append(-1)
                print("CreateError: id =", group[k][2])
                continue
        for k in range(len(group)):
            task_id = responses[k]
            if task_id == -1:   # create error
                continue
            try:
                result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
            except Exception:
                print("ResponseError: id =", group[k][2])
                continue
            task_status = result_response.task_status
            while task_status != 'SUCCESS' and task_status != 'FAILED':
                result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                # print(result_response)
                task_status = result_response.task_status
                time.sleep(1)
            glm_res = result_response.choices[0].message.content
            glm_ans = me.get_ans(glm_res)
            cor_ans = group[k][1]
            point = me.check_ans(glm_ans, cor_ans)
            correct_n += point
            total_n += 1
            print("id:", group[k][2], "glm_answer:", glm_ans, " correct_answer:", cor_ans,
                  " point:", point, " total_accuracy:", correct_n, "/", total_n, "=", correct_n / total_n)
            write_dict = {'id': group[k][2], 'prompt': group[k][0], 'content': glm_res, 'glm_answer': glm_ans, 'correct_answer': cor_ans}
            write_line = json.dumps(write_dict, ensure_ascii=False)
            write_lines.append(write_line+'\n')
        with open(write_file, 'a', encoding='utf-8') as rf:
            rf.writelines(write_lines)
        group = []
    if i == n:
        break
    s = ''
    for j in range(7):
        s += lines[i+j]
    json_data = json.loads(s)
    eid = json_data['id']
    q = json_data['original_text']
    cor_ans = json_data['ans']
    request = [q, cor_ans, eid]
    group.append(request)
    i += 7
