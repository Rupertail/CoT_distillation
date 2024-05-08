import json
import re

with open('normal_train_pure.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
write_file = 'normal_train_pure_selected.json'

pattern = r"答案是|答案为|正确选项是|正确选项为"

write_lines = []
for i in range(n):
    s = lines[i]
    json_data = json.loads(s)
    prompt = json_data['prompt']
    content = json_data['content']
    matches = re.findall(pattern, content)
    if len(matches) != 1:
        continue
    prefix = matches[0]
    index = content.find(prefix)
    sub_str = content[index:]
    matches = re.findall(r'[ABCD]', sub_str)
    if len(matches) != 1:
        continue
    alpha = matches[0]
    content = content[0: index] + "答案是" + alpha + "。"
    write_dict = {'prompt': prompt, 'content': content}
    write_line = json.dumps(write_dict, ensure_ascii=False)
    write_lines.append(write_line + '\n')

with open(write_file, 'a', encoding='utf-8') as wf:
    wf.writelines(write_lines)
print('finished')
