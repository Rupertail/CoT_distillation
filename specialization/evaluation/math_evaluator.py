import re
from fractions import Fraction


def get_ans(text: str):
    """extract the last int, float or fraction of response as model's answer."""
    pattern = r'\d+\.\d+%|\d+\.\d+|\d+/\d+|\d+%|frac{\d+}{\d+}|\d+'  # 匹配整数或浮点数或分数或百分数
    matches = re.findall(pattern, text)
    if len(matches) == 0:
        return None
    else:
        return matches[-1]


def check_ans(response: str, correct_ans: str):
    """check if model's answer equals to correct answer. RETURN: int(1) or int(0)"""
    if response is None:
        return int(0)
    if response.find('%') != -1:
        return check_ans(response[:-1], correct_ans) or check_ans(str(float(response[:-1])/100), correct_ans)
    if correct_ans.find('%') != -1:
        return check_ans(response, correct_ans[:-1]) or check_ans(response, str(float(correct_ans[:-1])/100))
    if response.find('frac') != -1:
        return check_ans(str(convert_frac(response)), correct_ans)
    if ('/' in response) and ('/' in correct_ans):
        # print(Fraction(response), convert_frac(correct_ans))
        if Fraction(response) == convert_frac(correct_ans):
            return int(1)
        else:
            return int(0)
    elif '/' in response:
        r = float(Fraction(response))
        c = float(correct_ans)
        if abs(r - c) < 0.05:
            return int(1)
        else:
            return int(0)
    elif '/' in correct_ans:
        r = float(response)
        c = float(convert_frac(correct_ans))
        if abs(r - c) < 0.05:
            return int(1)
        else:
            return int(0)
    else:
        r = float(response)
        c = float(correct_ans)
        if abs(r - c) < 0.05:
            return int(1)
        else:
            return int(0)


def convert_frac(s: str):
    """input fraction string such as '(a)/(b)' or 'frac{a}/{b}' or 'a((b)/(c)), RETURN: Fraction"""
    pattern = r'\d+'
    matches = re.findall(pattern, s)
    # print(matches)
    if len(matches) == 1:
        return Fraction(int(matches[0]), 1)
    elif len(matches) == 2:
        return Fraction(int(matches[0]), int(matches[1]))
    elif len(matches) == 3:
        a = int(matches[0])
        b = int(matches[1])
        c = int(matches[2])
        b += a * c
        return Fraction(b, c)
    else:
        return None
