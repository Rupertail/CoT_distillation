import json
from zhipuai import ZhipuAI
from specialization.evaluation import math_evaluator as me
import time

f = open('../zhipu_api_key.txt')
api_key = f.readline().strip()  # 读取第一行
client = ZhipuAI(api_key=api_key)  # 填写您自己的APIKey
# zero_shot_prompt = "作为一个 AI 助手，你的任务是帮助用户解决复杂的数学问题。对于每个问题，你需要独立解决它，并最终提供反馈。在这个过程中，请展示你的每一步推理过程。我有一个数学问题需要帮助,问题是："
zero_shot_prompt = "要求：对于下列问题，你需要独立解决它。在这个过程中，请展示你的每一步推理过程。为方便评测，回答的末尾需符合\"答案：<ans>\"的格式，其中<ans>为答案对应的数值，若结果为分数，需表示为a/b的形式。\n问题是："
write_file = "glm3-turbo_math_test_result_2.json"

with open('data/math23k_test.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
group_n = int(100)  # question number in a group
i = int(0)    # begin line
correct_n = int(0)
total_n = int(0)
group = []

while i <= n:
    if (i != 0) and ((i % (group_n * 7) == 0) or (i == n)):
        responses = []
        write_lines = []
        for k in range(len(group)):
            response = client.chat.asyncCompletions.create(
                model="glm-3-turbo",  # 填写需要调用的模型名称
                messages=[
                    {"role": "user", "content": zero_shot_prompt+group[k][0], "temperature": 0.97}
                ],
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

'''
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": "张明有120元钱，买书用去80%，买文具的钱是买书的15%．买文具用去多少元？"},
        {"role": "assistant", "content": "这个问题涉及到百分比的计算。让我们分步骤来解决它。\n\n首先，我们要确定张明买书用去了多少钱。\n\n1. 张明总共有120元钱。\n2. 他用去了80%来买书。\n\n计算张明买书的花费：\n120元 × 80% = 120元 × 0.8 = 96元\n\n所以，张明买书用去了96元。\n\n接下来，我们需要确定张明用去买文具的钱。题目中提到买文具的钱是买书的15%。\n\n计算张明买文具的花费：\n买书的钱 × 15% = 96元 × 0.15 = 14.4元\n\n因此，张明买文具用去了14.4元。"},
        {"role": "user", "content": "图书馆里有一些科技书和文艺书，其中科技书占(4/5)，如果用文艺书换走科技书20本，那么科技书占全部的(8/15)，问原来科技书有多少本？"}
    ],
)
print(response.choices[0].message.content)
'''
