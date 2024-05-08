import json
import specialization.evaluation.math_evaluator as me

with open('classified_prompts_original.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
write_file = 'classified_prompts.json'

write_lines = []
for i in range(n):
    s = lines[i]
    json_data = json.loads(s)
    prompt = json_data['prompt']
    content = json_data['content']
    glm_ans = json_data['glm_answer']
    cor_ans = json_data['correct_answer']
    point = me.check_ans(glm_ans, cor_ans)
    if point == 0:
        continue
    write_dict = {'prompt': prompt, 'content': content}
    write_line = json.dumps(write_dict, ensure_ascii=False)
    write_lines.append(write_line + "\n")

with open(write_file, 'a', encoding='utf-8') as wf:
    wf.writelines(write_lines)
print('finished')
