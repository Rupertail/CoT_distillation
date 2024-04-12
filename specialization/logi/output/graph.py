import matplotlib.pyplot as plt
import numpy as np
import json

with open('trainer_state.json', 'r', encoding='utf-8')as fp:
    lines = fp.readlines()
n = len(lines)
size = n / 6

x_axis_data = []  # x
y_axis_data = []  # y

i = 0
s = ''
while i < n:
    if (i > 0) and ((i % 6) == 0):
        json_data = json.loads(s[:-1])
        step = json_data['step']
        loss = json_data['loss']
        x_axis_data.append(step)
        y_axis_data.append(loss)
        s = ''
    s += lines[i].strip()
    i += 1

plt.plot(x_axis_data, y_axis_data, 'bo--', alpha=0.5, linewidth=1, label='1e-2')  # 'bo-'表示蓝色实线，数据点实心原点标注
# plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

plt.legend()  # 显示上面的label
plt.xlabel('step')  # x_label
plt.ylabel('loss')  # y_label

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
