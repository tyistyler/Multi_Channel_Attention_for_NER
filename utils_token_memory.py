# coding    : utf-8
# @Time     : 2020/12/24 16:27
# @Author   : tyler
# @File     : utils_token_memory.py
# @Software : PyCharm
import re
rNUM = '(-|\+)?\d+((\.)\d+)?%?'
rENG = '[A-Za-z_.]+'
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  #类似空格
            inside_code = 32  #空格
        elif (inside_code >= 65281 and inside_code <= 65374):#全、半角转换
            # print('inside_code',inside_code)
            inside_code -= 65248
            # print('inside_code_new',inside_code)
        rstring += chr(inside_code)
    return rstring
def make_lexicon(src, tgt):
    f = open(src, 'r', encoding='utf-8')
    words2id = {}
    id2words = {}
    lines = f.readlines()
    # lines = ['这  部  电视剧  真实  生动  地  表现  了  １９４８年６—７月  间  ，  在  刘伯承  、  邓小平  同志  的  精心  运筹  指挥  下  ，  我军  发起  襄阳  战役  。']
    for line in lines:
        # words = line.strip().split()
        words = strQ2B(line).split()
        words = [word for word in words if len(word) > 1 and len(word)<6]  # 限制词语长度
        # print(words)
        for word in words:
            if bool(re.search(r'\d', word)):   # 判断当前字符串是否存在数字
                continue
            elif bool(re.search(r'[a-zA-Z]+',word)):
                continue
            else:
                if word not in words2id:
                    words2id[word] = len(words2id)
                    id2words[len(id2words)] = word

        # break
    #     words = [word for word in words if len(word)>1 and len(word)<6] # 限制词语长度
    #     words = words   # 过滤数字

    print(len(words2id))
    g = open(tgt, 'w', encoding='utf-8')
    for w,n in words2id.items():
        g.write(w + '\n')
    g.close()
    f.close()
    return words2id, id2words

if __name__ == '__main__':
    src = 'data/msr/msr_training.utf8'
    tgt = 'memory_lexicon/msr_lexicon.txt'
    words2id, id2words = make_lexicon(src, tgt)
    # a = 'aa1'
    # if re.search(r'[a-zA-Z]+',a):
    #     print(a)

