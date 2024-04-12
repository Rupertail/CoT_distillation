import json
from zhipuai import ZhipuAI
from specialization.evaluation import math_evaluator as me
import time
from config import fs_prompt, get_equation

f = open('../zhipu_api_key.txt')
api_key = f.readline().strip()  # 读取第一行
client = ZhipuAI(api_key=api_key)  # 填写您自己的APIKey
write_file = "18_prompt_generation.json"
valid_file = "valid_prompts.json"
fs = fs_prompt()
equations = get_equation('math23k_train.json')
zero_shot_prompt = "要求：对于下列问题，你需要独立解决它。在这个过程中，请展示你的每一步推理过程。为方便评测，回答的末尾需符合\"答案：<ans>\"的格式，其中<ans>为答案对应的数值，如\"1\"、\"1.5\"、\"1/2\"、\"1%\"等。\n\n问题是："

with open('./17_prompt_generation.json', 'r', encoding='utf-8') as rp:
    lines = rp.readlines()
n = len(lines)
group_n = int(100)  # question number in a group
i = int(0)    # begin line
j = int(0)
correct_n = int(0)
total_n = int(0)
group = []

while i <= n:
    if (i != 0) and ((j % group_n == 0) or (i == n)):
        responses = []
        write_lines = []
        valid_lines = []
        for k in range(len(group)):
            """
            messages = [
                           {"role": "user", "content": fs['q1']},
                           {"role": "assistant", "content": fs['a1']},
                           {"role": "user", "content": fs['q2']},
                           {"role": "assistant", "content": fs['a2']},
                           {"role": "user", "content": fs['q3']},
                           {"role": "assistant", "content": fs['a3']},
                           {"role": "user", "content": fs['q4']},
                           {"role": "assistant", "content": fs['a4']},
                           {"role": "user", "content": fs['q5']},
                           {"role": "assistant", "content": fs['a5']},
                           {"role": "user", "content": "问题：" + group[k][0], "temperature": 0.97}
                       ]
            """
            equ = equations[group[k][2]]
            extend = "\n\n提示：" + equ
            messages = [
                           {"role": "user", "content": zero_shot_prompt + group[k][0], "temperature": 0.97}
                       ]
            response = client.chat.asyncCompletions.create(
                model="glm-3-turbo",  # 填写需要调用的模型名称
                messages=messages
            )
            responses.append(response.id)
        for k in range(len(group)):
            task_id = responses[k]
            result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
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
            write_dict = {'id': group[k][2], 'prompt': group[k][0], 'content': glm_res, 'glm_answer': glm_ans, 'correct_answer': cor_ans, 'point': point}
            write_line = json.dumps(write_dict, ensure_ascii=False)
            write_lines.append(write_line+'\n')
            if point == 1:
                valid_lines.append(write_line+'\n')
        with open(write_file, 'a', encoding='utf-8') as wf:
            wf.writelines(write_lines)
        with open(valid_file, 'a', encoding='utf-8') as vf:
            vf.writelines(valid_lines)
        group = []
    if i == n:
        break
    s = lines[i]
    json_data = json.loads(s)
    point = json_data['point']
    if int(point) == 0:
        eid = json_data['id']
        q = json_data['prompt']
        cor_ans = json_data['correct_answer']
        request = [q, cor_ans, eid]
        group.append(request)
        j += 1
    i += 1
