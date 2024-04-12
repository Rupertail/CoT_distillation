import re

def get_ans(text: str):
    """extract option from model's answer text"""
    sm = special_match(text)
    if sm is not None:
        return sm
    return common_match(text)


def special_match(text: str):
    """extract [A-D] which is after a special prefix"""
    pattern = r'答案是[A-D]|答案：[A-D]|答案为[A-D]|正确选项是[A-D]|正确选项：[A-D]|正确选项为[A-D]|答案是:[A-D]|答案为:[A-D]|正确选项是:[A-D]|正确选项为:[A-D]'
    matches = re.findall(pattern, remove_spaces(text))
    if len(matches) == 0:
        return None
    else:
        return common_match(matches[-1])


def common_match(text: str):
    """extract the last [A-D]"""
    pattern = r'[ABCD]'
    matches = re.findall(pattern, text)
    if len(matches) == 0:
        return None
    else:
        return matches[-1]


def check_ans(response: str, correct_ans: str):
    """check if model's answer equals to correct answer. RETURN: int(1) or int(0)"""
    if response is None:
        return int(0)
    elif response == correct_ans:
        return int(1)
    else:
        return int(0)


def remove_spaces(string: str):
    return "".join(string.split())
