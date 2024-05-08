import matplotlib.pyplot as plt
import numpy as np
import json

with open('trainer_state_1.json', 'r', encoding='utf-8')as fp:
    lines1 = fp.readlines()
n1 = len(lines1)
size1 = n1 / 6

with open('trainer_state_2.json', 'r', encoding='utf-8')as fp:
    lines2 = fp.readlines()
n2 = len(lines2)
size2 = n2 / 6

with open('trainer_state_3.json', 'r', encoding='utf-8')as fp:
    lines3 = fp.readlines()
n3 = len(lines3)
size3 = n3 / 6

x_axis_data1 = []  # x1
y_axis_data1 = []  # y1
x_axis_data2 = []  # x2
y_axis_data2 = []  # y2
x_axis_data3 = []  # x3
y_axis_data3 = []  # y3

i1 = 0
s1 = ''
while i1 < n1:
    if (i1 > 0) and ((i1 % 6) == 0):
        json_data = json.loads(s1[:-1])
        step = json_data['step']
        loss = json_data['loss']
        x_axis_data1.append(step)
        y_axis_data1.append(loss)
        s1 = ''
    s1 += lines1[i1].strip()
    i1 += 1

i2 = 0
s2 = ''
while i2 < n2:
    if (i2 > 0) and ((i2 % 6) == 0):
        json_data = json.loads(s2[:-1])
        step = json_data['step']
        loss = json_data['loss']
        x_axis_data2.append(step)
        y_axis_data2.append(loss)
        s2 = ''
    s2 += lines2[i2].strip()
    i2 += 1

i3 = 0
s3 = ''
while i3 < n3:
    if (i3 > 0) and ((i3 % 6) == 0):
        json_data = json.loads(s3[:-1])
        step = json_data['step']
        loss = json_data['loss']
        x_axis_data3.append(step)
        y_axis_data3.append(loss)
        s3 = ''
    s3 += lines3[i3].strip()
    i3 += 1

plt.plot(x_axis_data1, y_axis_data1, 'g-', alpha=0.5, linewidth=1, label='2e-4')  # 'bo-'表示蓝色实线，数据点实心原点标注
plt.plot(x_axis_data2, y_axis_data2, 'b-', alpha=0.5, linewidth=1, label='5e-4')
plt.plot(x_axis_data3, y_axis_data3, 'r-', alpha=0.5, linewidth=1, label='2e-3')
# plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

plt.legend()  # 显示上面的label
plt.xlabel('step')  # x_label
plt.ylabel('loss')  # y_label

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
