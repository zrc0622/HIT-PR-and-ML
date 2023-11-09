import numpy as np
import jieba
from torchtext.data import Field, TabularDataset

def tokenize_cn(text):
    return list(jieba.cut(text)) # 中文分词

TEXT = Field(sequential=True, tokenize=tokenize_cn)  # 定义文本字段，包含分词和小写转换
LABEL = Field(sequential=False, use_vocab=False)  # 定义标签字段，use_vocab不使用词汇表

datafields = [('category', LABEL), ('sentiment', LABEL), ('comment', TEXT)]  # 定义数据字段：类别、情感、评论

train_data, test_data, valid_data = TabularDataset.splits(
                            path='../data/online_shopping_10_cats', train='train_data.csv', test='test_data.csv', validation='validation_data.csv',
                            format='csv', skip_header=True, fields=datafields)  

TEXT.build_vocab(train_data, test_data, valid_data)

num_words = len(TEXT.vocab)  # 获取词汇表中单词的数量
embedding_dim = 100  # 设置词向量维度

# 创建一个包含所有token的随机词向量矩阵
random_embeddings = np.random.rand(len(TEXT.vocab), embedding_dim)

# 将随机词向量保存到文件
with open('./.vector_cache/random_embeddings.txt', 'w', encoding='utf-8') as file:
    for token, idx in TEXT.vocab.stoi.items():
        word = token
        vector = ' '.join(map(str, random_embeddings[idx]))
        if word != ' ':
            file.write(f'{word} {vector}\n')

