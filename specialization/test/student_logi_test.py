import requests
import json
import specialization.evaluation.logi_evaluator as le
import specialization.logi.ptuning.api as ft_api

FINE_TUNED = 0
url = "http://127.0.0.1:8000"
with open('data/logiqa_test.txt', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
i = int(0)
'''
h = [["要求：阅读文本，回答问题，从A、B、C、D四个选项中选择符合问题描述的答案。\n文本：只有干部学法，我不是干部，不用学法。\n问题：哪个以下是与上述推理结构最相似的一个？\n选项：A:考不上就考不上南大，张进了南大，所以考上了。\nB:只要通过考试，就可以考上南京大学。小张考试不及格，进不了南京大学。\nC:只有通过考试，才能考上南京大学。南京大学。张考试不及格，进不了南京大学。\nD:只有通过考试才能进入南京大学。张考试通过了，才能进南京大学学习.",
      "答案是C。\n\n只有干部学法，即干部是学法的前提；我不是干部，则不满足前提，因此推导出不用学法。C选项中通过考试是考上南京大学的前提，张考试不及格，说明没有通过考试，则不满足前提，因此推导出不能考上南京大学。结构与文本相符，所以答案是C。"]]
'''
write_file = "glm2_logi_test_result_pure.json"

if FINE_TUNED == 1:
    tokenizer, model = ft_api.load_model('normal', 500)

correct_n = int(0)
total_n = int(0)
for line in lines:
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
    # p = "文本：" + t + "\n\n问题：" + q + "\n\n选项：" + "\nA:" + o1 + "\nB:" + o2 + "\nC:" + o3 + "\nD:" + o4 + "\n\n请一步一步进行逻辑推理。"
    # None-prompt
    p = "文本：" + t + "\n\n问题：" + q + "\n\n选项：" + "\nA:" + o1 + "\nB:" + o2 + "\nC:" + o3 + "\nD:" + o4
    d = {"prompt": p, "history": []}
    if FINE_TUNED == 0:
        r = requests.post(url, data=json.dumps(d))
        res_content = json.loads(r.text)['response']
    else:
        r, h = model.chat(tokenizer, p, history=[])
        # print("r:", r)
        res_content = r
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
