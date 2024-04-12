import json

with open('valid_prompts.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
write_file = 'normal_train.json'

write_lines = []
for i in range(n):
    s = lines[i]
    json_data = json.loads(s)
    prompt = json_data['prompt']
    content = json_data['content']
    point = json_data['point']
    if point == 1 and len(prompt) + len(content) < 1500:
        write_dict = {'prompt': prompt, 'content': content}
        write_line = json.dumps(write_dict, ensure_ascii=False)
        write_lines.append(write_line + '\n')
with open(write_file, 'a', encoding='utf-8') as wf:
    wf.writelines(write_lines)
print('finished')
