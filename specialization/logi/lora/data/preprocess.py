import json

with open('logiQA_normal_pure_raw.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
write_file = 'logiQA_normal_pure.json'

write_lines = ['[\n']
for i in range(n):
    s = lines[i]
    json_data = json.loads(s)
    prompt = json_data['prompt']
    content = json_data['content']
    write_dict = {'instruction': prompt, 'input': '', 'output': content, 'history': []}
    write_line = json.dumps(write_dict, ensure_ascii=False)
    write_lines.append(write_line + ',\n')  # 需手动去除最后一行的','
write_lines.append(']')

with open(write_file, 'a', encoding='utf-8') as wf:
    wf.writelines(write_lines)
print('finished')
