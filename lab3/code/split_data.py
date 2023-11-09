import pandas as pd
import re

# 对数据集进行划分
data = pd.read_csv('../data/online_shopping_10_cats/online_shopping_10_cats.csv')

def remove_punctuation(line): #使用正则表达式来过滤各种标点符号
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line


# 映射字典
category_mapping = {
    '书籍': 0,
    '平板': 1,
    '手机': 2,
    '水果': 3,
    '洗发水': 4,
    '热水器': 5,
    '蒙牛': 6,
    '衣服': 7,
    '计算机': 8,
    '酒店': 9
}

# 将 'cat' 列映射为整数标签
data['cat'] = data['cat'].map(category_mapping)
data['review'] = data['review'].apply(lambda x: remove_punctuation(x))

# 划分数据集索引
train_indices = [i for i in range(1, data.shape[0]) if i % 5 not in [0, 4]]  # 训练集索引
validation_indices = [i for i in range(1, data.shape[0]) if i % 5 == 4]  # 验证集索引
test_indices = [i for i in range(1, data.shape[0]) if i % 5 == 0]  # 测试集索引

# 使用索引划分数据集
train_data = data.iloc[train_indices]  # 训练集
validation_data = data.iloc[validation_indices]  # 验证集
test_data = data.iloc[test_indices]  # 测试集

# 保存划分后的数据集为新的CSV文件
train_data.to_csv('../data/online_shopping_10_cats/train_data.csv', index=False)
validation_data.to_csv('../data/online_shopping_10_cats/validation_data.csv', index=False)
test_data.to_csv('../data/online_shopping_10_cats/test_data.csv', index=False)
