import requests
import json
import specialization.evaluation.logi_evaluator as le
import time

# original url:
url = "http://127.0.0.1:8000/v1/chat/completions"
with open('data/logiqa_test.txt', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
i = int(0)
write_file = "glm2_logi_test_result_lora_normal_5e-5-2500.json"

# print(requests.get("http://127.0.0.1:8000/v1/models").text)
start_time = time.time()

correct_n = int(0)
total_n = int(0)
while i < n:
    line = lines[i]
    json_data = json.loads(line)
    q = json_data['question']
    o = json_data['options']
    a = json_data['answer']
    t = json_data['text']
    eid = json_data['example_id']
    o1 = o[0]
    o2 = o[1]
    o3 = o[2]
    o4 = o[3]
    '''p = "要求：阅读文本，回答问题，从A、B、C、D四个选项中选择符合问题描述的答案。\n文本：" + t + "\n问题：" + q + "\n选项：" + \
        "\nA:" + o1 + "\nB:" + o2 + "\nC:" + o3 + "\nD:" + o4'''
    # zs-CoT-1:
    '''p = "要求：阅读文本和问题，一步一步仔细思考，从A、B、C、D四个选项中选择符合问题描述的答案。\n\n文本：" + t + "\n\n问题：" + q + "\n\n选项：" + \
        "\nA:" + o1 + "\nB:" + o2 + "\nC:" + o3 + "\nD:" + o4'''
    # zs-CoT-2:
    p = "文本：" + t + "\n\n问题：" + q + "\n\n选项：" + "\nA:" + o1 + "\nB:" + o2 + "\nC:" + o3 + "\nD:" + o4 + "\n\n请一步一步进行逻辑推理。"
    # None-prompt
    # p = "文本：" + t + "\n\n问题：" + q + "\n\n选项：" + "\nA:" + o1 + "\nB:" + o2 + "\nC:" + o3 + "\nD:" + o4
    messages = [{"role": "user", "content": p}]
    d = {"model": "chatglm", "messages": messages}
    # d = {"messages": messages}
    # print("d:", d)
    r = requests.post(url, data=json.dumps(d))
    # print("r:", r.text)
    res_content = json.loads(r.text)["choices"][0]["message"]["content"]
    glm_ans = le.get_ans(res_content)
    cor_ans = chr(ord('A') + int(a))
    point = le.check_ans(glm_ans, cor_ans)
    correct_n += point
    total_n += 1
    print("id:", eid, "glm_answer:", glm_ans, " correct_answer:", cor_ans,
          " point:", point, " total_accuracy:", correct_n, "/", total_n, "=", correct_n / total_n)
    # print(r.text)
    write_dict = {'id': eid, 'prompt': p, 'content': res_content, 'glm_answer': glm_ans,
                  'correct_answer': cor_ans}
    write_line = json.dumps(write_dict, ensure_ascii=False)
    with open(write_file, 'a', encoding='utf-8') as rf:
        rf.write(write_line + '\n')
    i += 1

end_time = time.time()
used_time = end_time - start_time
print("totally used time: {.2f}s\n".format(used_time))
print("used time per instruction: {.2f}s\n".format(used_time/total_n))
print("Results saved in \'" + write_file + "\'.")
