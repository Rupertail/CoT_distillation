import json
import specialization.evaluation.logi_evaluator as le
import specialization.evaluation.math_evaluator as me

with open('glm2_math_test_result_lora_normal_1e-3-13000.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)

valid = int(0)
for i in range(n):
    s = lines[i]
    json_data = json.loads(s)
    glm_ans = json_data['glm_answer']
    cor_ans = json_data['correct_answer']
    valid += le.check_ans(glm_ans, cor_ans)
    # valid += me.check_ans(glm_ans, cor_ans)

print("accuracy:", valid, "/", n, "=", valid / n)
