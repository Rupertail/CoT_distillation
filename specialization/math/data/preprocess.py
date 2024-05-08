import json

with open('classified_prompts_frac_cleaned.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
write_file = 'math23k_classified.json'

write_lines = ['[\n']
for i in range(n):
    # print(i)
    s = lines[i]
    json_data = json.loads(s)
    prompt = json_data['prompt']
    content = json_data['content']
    if len(prompt+content) > 1500 or content.find("错误") != -1 or content.find("有误") != -1:
        continue
    write_dict = {'instruction': prompt, 'input': '', 'output': content, 'history': []}
    write_line = json.dumps(write_dict, ensure_ascii=False)
    write_lines.append(write_line + ',\n')  # 需手动去除最后一行的','
write_lines.append(']')

with open(write_file, 'a', encoding='utf-8') as wf:
    wf.writelines(write_lines)
print('finished')
