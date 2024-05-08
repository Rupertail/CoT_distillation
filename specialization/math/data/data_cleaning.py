import json
import re


def main():
    with open('classified_prompts_text_cleaned.json', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    n = len(lines)
    i = int(0)

    write_file = "classified_prompts_frac_cleaned.json"

    while i < n:
        line = lines[i]
        json_data = json.loads(line)
        p = json_data['prompt']
        c = json_data['content']
        # c = clean_text(c)
        c = clean_frac(c)
        write_dict = {'prompt': p, 'content': c}
        write_line = json.dumps(write_dict, ensure_ascii=False)
        with open(write_file, 'a', encoding='utf-8') as rf:
            rf.write(write_line + '\n')
        i += 1

    print("finished!")


def clean_text(raw: str):
    """\\text{a} -> a"""
    s = raw
    while s.find(r"\text") != -1:
        beg = s.find(r"\text")
        end = matched_brace(s, beg)
        s = s[0: beg] + s[beg+6: end] + s[end+1:]
    return s


def clean_frac(raw: str):
    """\\frac{a}{b} -> (a/b)"""
    s = raw
    while s.find(r"\frac") != -1:
        begin = s.find(r"\frac")
        l1 = begin + 5
        r1 = matched_brace(s, l1)
        l2 = r1 + 1
        r2 = matched_brace(s, l2)
        s = s[0: begin] + '(' + s[l1+1: r1] + '/' + s[l2+1: r2] + ')' + s[r2+1:]
    return s


def matched_brace(s: str, index: int):
    """input str and index of left brace \'{\' , return index of matched right brace \'}\'."""
    i = index
    level = 0
    while i < len(s):
        if s[i] == '{':
            level += 1
        elif s[i] == '}':
            level -= 1
            if level == 0:
                return i
        i += 1
    return -1


if __name__ == '__main__':
    main()
