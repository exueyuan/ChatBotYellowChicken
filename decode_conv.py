
import os
import re
import sys
import sqlite3
from collections import Counter

from tqdm import tqdm


def file_lines(file_path):
    # 打开文件，读取所有数据
    with open(file_path, 'rb') as fp:
        b = fp.read()
    # 转成utf-8的格式
    content = b.decode('utf8', 'ignore')
    lines = []
    # tqdm：进度条，每行数据都进行进度条加载
    for line in tqdm(content.split('\n')):
        try:
            line = line.replace('\n', '').strip()
            if line.startswith('E'):  # 起始符号是否为E
                lines.append('')
            elif line.startswith('M '):
                chars = line[2:].split('/')
                while len(chars) and chars[len(chars) - 1] == '.':
                    chars.pop()
                if chars:
                    sentence = ''.join(chars)
                    # 把空格替换成'，'
                    sentence = re.sub('\s+', '，', sentence)
                    lines.append(sentence)
        except:
            lines.append('')
    return lines

def contain_chinese(s):
    # 是否包含中文
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False


# 验证
def valid(a, max_len=0):
    if len(a) > 0 and contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:  # 防止验证太长
            return True
    return False

def insert(a, b, cur):
    a_ = a.replace("'", "''")
    b_ = b.replace("'", "''")
    sql = """
    INSERT INTO conversation (ask, answer) VALUES
    ('{}', '{}')
    """.format(a_, b_)
    cur.execute(sql)

def insert_if(question, answer, cur, input_len=500, output_len=500):
    if valid(question, input_len) and valid(answer, output_len):
        insert(question, answer, cur)
        return 1
    return 0

def main(file_path: str):
    # 获取每行的数据
    lines = file_lines(file_path)

    print('一共读取 %d 行数据' % len(lines))

    db = 'db/conversation.db'
    if os.path.exists(db):
        # 如果存在数据库，那就把数据库删除掉
        os.remove(db)
    if not os.path.exists(os.path.dirname(db)):
        # 如果不存在数据库的文件夹，那么创建文件夹
        os.makedirs(os.path.dirname(db))
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    # 执行sql语句创建数据库
    cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation
            (ask text, answer text);
            """)
    conn.commit()

    words = Counter()
    answer = ''

    inserted = 0

    for index, line in tqdm(enumerate(lines), total=len(lines)):
        # 创建词库
        words.update(Counter(line))
        # 生成问答对，把上一次的回答作为当前的提问
        '''a = b
        b = line
        ask = a
        answer = b'''
        ask = answer
        answer = line

        inserted += insert_if(ask, answer, cur)
        # 批量提交
        if inserted != 0 and inserted % 50000 == 0:
            conn.commit()

    conn.commit()


if __name__ == '__main__':
    file_path = "db/xiaohuangji50w_fenciA.conv"
    main(file_path)
