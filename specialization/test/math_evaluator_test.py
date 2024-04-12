import json
from specialization.evaluation import math_evaluator as me

with open('data/math23k_test.json', 'r', encoding='utf8')as ds:
    lines = ds.readlines()
n = len(lines)

i = int(0)
ca_list = []
point = int(0)
while i < n:
    s = ''
    for j in range(7):
        s += lines[i+j]
    json_data = json.loads(s)
    correct_ans = json_data['ans']
    eid = json_data['id']
    add = me.check_ans("99.9%", correct_ans)
    point += add
    if add == 1:
        print("eid:", eid, "correct_ans:", correct_ans, "point:", point)
    i += 7
