import json

import zhipuai
from zhipuai import ZhipuAI
from specialization.evaluation import logi_evaluator as le
import time

f = open('../zhipu_api_key.txt')
api_key = f.readline().strip()  # 读取第一行
client = ZhipuAI(api_key=api_key)  # 填写您自己的APIKey
write_file = "classified_prompts.json"
error_list = "logiqa_classified_error_list.txt"

kid = "1775143931248513024"
k_prompt = "阅读文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n，找到问题\n\"\"\"\n{{question}}\n\"\"\"\n的类型和解题思路和方法，并用思维链一步一步推理并解决问题。" \
        "\n\n\n输出格式：\n【类型】\n\n直接输出问题的类型。\n\n【解题思路和方法】\n\n根据文档内容总结该类型问题的一般性解题思路和方法。" \
        "\n\n【解题步骤】\n\n根据【解题思路和方法】以及文档中的例题，用思维链一步一步推理并解决问题。\n\n\n不要复述问题，直接开始回答。"

with open('data/logiqa_train.txt', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
group_n = int(50)  # question number in a group
i = int(0)    # begin line
correct_n = int(0)
total_n = int(0)
group = []

while i <= n:
    if (i != 0) and ((i % group_n == 0) or (i == n)):
        responses = []
        write_lines = []
        for k in range(len(group)):
            response = client.chat.asyncCompletions.create(
                model="glm-4",  # 填写需要调用的模型名称
                messages=[
                    {"role": "user", "content": group[k][0], "temperature": 0.6}
                ],
                tools=[
                    {
                        "type": "retrieval",
                        "retrieval": {
                            "knowledge_id": kid,
                            "prompt_template": k_prompt
                        }
                    }
                ],
            )
            responses.append(response.id)
        for k in range(len(group)):
            task_id = responses[k]
            try:
                result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
            except zhipuai.core._errors.APIRequestFailedError:
                error_info = "line: " + str(i-len(group)+k+1) + "\tid: " + str(group[k][2]) + "\n"
                with open(error_list, 'a', encoding='utf-8') as el:
                    el.write(error_info)
                continue
            task_status = result_response.task_status
            while task_status != 'SUCCESS' and task_status != 'FAILED':
                result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                # print(result_response)
                task_status = result_response.task_status
                time.sleep(1)
            glm_res = result_response.choices[0].message.content
            glm_ans = le.get_ans(glm_res)
            cor_ans = group[k][1]
            point = le.check_ans(glm_ans, cor_ans)
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
    s = lines[i]
    json_data = json.loads(s)
    eid = json_data['example_id']
    q = json_data['question']
    a = json_data['answer']
    t = json_data['text']
    o = json_data['options']
    o1 = o[0]
    o2 = o[1]
    o3 = o[2]
    o4 = o[3]
    prompt = "文本：" + t + "\n\n问题：" + q + "\n\n选项：" + \
        "\nA:" + o1 + "\nB:" + o2 + "\nC:" + o3 + "\nD:" + o4
    cor_ans = chr(ord('A') + int(a))
    request = [prompt, cor_ans, eid]
    group.append(request)
    i += 1
