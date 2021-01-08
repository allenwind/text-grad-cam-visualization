import re
import glob
import random
import itertools
import collections
import pandas as pd
import numpy as np

_THUCNews = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"
def load_THUCNews_title_label(file=_THUCNews, nobrackets=True):
    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    np.random.shuffle(lines)
    titles = []
    labels = []
    for line in lines:
        title, label = line.split("\t")
        if not title:
            continue

        # 去掉括号内容
        if nobrackets:
            title = re.sub("\(.+?\)", lambda x:"", title)
        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label:i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

_THUContent = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
def load_THUCNews_content_label(file=_THUContent, shuffle=True):
    categoricals = {'体育': 0, '娱乐': 1, '家居': 2, '彩票': 3, 
                    '房产': 4, '教育': 5, '时尚': 6, '时政': 7, 
                    '星座': 8, '游戏': 9, '社会': 10, '科技': 11, 
                    '股票': 12, '财经': 13}
    
    files = glob.glob(file)
    if shuffle:
        random.shuffle(files)

    def Xy_generator(files):
        for path in files:
            label = path.rsplit("/", -2)[-2]
            with open(path, encoding="utf-8") as fd:
                _ = fd.readline() # skip title
                content = fd.read().strip()
            content = content.replace("\n", "").replace("\u3000", "")
            yield content, categoricals[label]
    return Xy_generator, files, categoricals

_HOTEL = "/home/zhiwen/workspace/dataset/classification/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
def load_hotel_comment(file=_HOTEL):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()[1:]
    random.shuffle(lines)
    X = []
    y = []
    for line in lines:
        if not line:
            continue
        label, commet = line.strip().split(",", 1)
        X.append(commet)
        y.append(int(label))
    categoricals = {"负面":0, "正面":1}
    return X, y, categoricals

_w100k = "/home/zhiwen/workspace/dataset/classification/weibo_senti_100k/weibo_senti_100k.csv"
def load_weibo_senti_100k(file=_w100k, noe=False):
    df = pd.read_csv(file)
    X = df.review.to_list()
    y = df.label.to_list()
    # 去 emoji 表情，提升样本训练难度
    if noe:
        X = [re.sub("\[.+?\]", lambda x:"", s) for s in X]
    categoricals = {"负面":0, "正面":1}
    return X, y, categoricals

_MOODS = "/home/zhiwen/workspace/dataset/classification/simplifyweibo_4_moods.csv"
def load_simplifyweibo_4_moods(file=_MOODS):
    df = pd.read_csv(file)
    X = df.review.to_list()
    y = df.label.to_list()
    categoricals = {"喜悦":0, "愤怒":1, "厌恶":2, "低落":3}
    return X, y, categoricals

_LCQMC = "/home/zhiwen/workspace/dataset/LCQMC/totals.txt"
def load_lcqmc(file=_LCQMC):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        x1, x2, label = line.strip().split("\t")
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

class SimpleTokenizer:
    """字转ID
    """

    def __init__(self, min_freq=16):
        self.char2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.min_freq = min_freq

    def fit(self, X):
        # 建立词ID映射表
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1
        # 过滤低频词
        chars = {i:j for i, j in chars.items() if j >= self.min_freq}
        # 0:MASK
        # 1:UNK
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNKNOW))
            ids.append(s)
        return ids

    def fit_transform(self, X):
        self.fit(X)
        ids = self.transform(X)
        return ids

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

    @property
    def vocab(self):
        return self.char2id

    def save(self, file):
        pass

    @classmethod
    def load(cls, file):
        pass

def find_best_maxlen(X, mode="max"):
    # 获取适合的截断长度
    ls = [len(sample) for sample in X]
    if mode == "mode":
        maxlen = np.argmax(np.bincount(ls))
    if mode == "mean":
        maxlen = np.mean(ls)
    if mode == "median":
        maxlen = np.median(ls)
    if mode == "max":
        maxlen = np.max(ls)
    return int(maxlen)

def get_class_weight(y):
    y = np.array(y)
    total = len(y)
    pos = y[y==1].size
    neg = y[y==0].size
    weight_for_0 = (1 / neg) * (total) / 2.0 
    weight_for_1 = (1 / pos) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight
