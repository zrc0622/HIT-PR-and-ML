import jieba
from torchtext.data import Field, TabularDataset, Iterator
from torchtext.vocab import Vectors, GloVe
from model.RNN import RNN
from model.LSTM import LSTM, BiLSTM
from model.GRU import GRU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import numpy as np
from plot import plot_matrix
import sys
sys.path.append(r'./model')


def main(**kwargs):

    # 超参数及路径
    model_type = kwargs.get('model')
    batch_size = 64
    embedding_dim = 100
    lr = kwargs.get('lr')
    print(f'learning rate is: {lr}')
    epochs = 0
    num_classes = 10
    hidden_size = 128
    model_dir = '/home/lsy/projects/lstm/code/log/08-19-31-48-RNN-0.001/model_0.1632149115819659_0.10482307303342342.pth'

    current_time = datetime.now()
    log_dir = current_time.strftime("./log/classify-%d-%H-%M-%S" + f"-{model_type}" + f"-{lr}") 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Folder '{log_dir}' created successfully.")
    else:
        print(f"Folder '{log_dir}' already exists.")
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize_cn(text):
        return list(jieba.cut(text)) # 中文分词

    TEXT = Field(sequential=True, tokenize=tokenize_cn)  # 定义文本字段，包含分词和小写转换
    LABEL = Field(sequential=False, use_vocab=False)  # 定义标签字段，use_vocab不使用词汇表

    datafields = [('cat', LABEL), ('label', LABEL), ('review', TEXT)]  # 定义数据字段：类别、情感、评论

    train_data, test_data, valid_data = TabularDataset.splits(
                                path='../data/online_shopping_10_cats', train='train_data.csv', test='test_data.csv', validation='validation_data.csv',
                                format='csv', skip_header=True, fields=datafields)  # 加载训练、测试和验证数据集，加载方式同datafields一致

    # embeddings = GloVe(name='6B', dim=embedding_dim) # 使用预训练词嵌入

    embeddings = Vectors(name='random_embeddings.txt') # 使用随机词向量

    TEXT.build_vocab(train_data, test_data, valid_data, vectors=embeddings)  # 构建文本字段的词汇表，并使用词嵌入
    print(TEXT.vocab.itos[1000]) # 输出单词
    print(TEXT.vocab.vectors[1000]) # 输出词向量
    print(f'max length of comments is: {len(TEXT.vocab)}')

    train_iterator, valid_iterator, test_iterator = Iterator.splits(
        (train_data, valid_data, test_data),
        batch_sizes=(batch_size, batch_size, batch_size),
        sort_key=lambda x: len(x.review),
        repeat=False)  # 创建训练、验证和测试数据的迭代器；sort_key对comment长度进行递增排序，以使模型训练更有效率
    if model_type == 'LSTM':
        model = LSTM(input_size=embedding_dim, hidden_size=hidden_size, output_size=num_classes)
        print(f'model is {model}')
    elif model_type == 'RNN':
        model = RNN(input_size=embedding_dim, hidden_size=hidden_size, output_size=num_classes)
        print(f'model is {model}')
    elif model_type == 'GRU':
        model = GRU(input_size=embedding_dim, hidden_size=hidden_size, output_size=num_classes)
        print(f'model is {model}')
    elif model_type == 'BiLSTM':
        model = BiLSTM(input_size=embedding_dim, hidden_size=hidden_size, output_size=num_classes)
        print(f'model is {model}')
    # model = torch.load(model_dir)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_iterator:
            optimizer.zero_grad()
            text, label = batch.review, batch.cat # (random, 64),(64)  random为最长序列的长度，存的是token的索引
            word_vectors = TEXT.vocab.vectors[text] # (random, 64, 100)  通过token的索引找到词向量
            # cat_onehot = nn.functional.one_hot(label, num_classes=num_classes) # (64, 10)  使用one hot将整数label转为十类
            
            word_vectors = TEXT.vocab.vectors[text].to(device)  
            label = label.to(device)
            hidden = model.init_hidden(word_vectors.size(1)).to(device)

            output, _ = model(word_vectors, hidden)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        writer.add_scalar('Loss/train', loss, epoch)
        
        if (epoch + 1) % 5 == 0:
            model.eval()  # 将模型设为评估模式

            total_correct = 0
            total_samples = 0

            with torch.no_grad():  # 在验证集或测试集上不进行梯度计算
                for batch in train_iterator:  # 使用测试迭代器（或验证迭代器）进行迭代
                    text, label = batch.review, batch.cat
                    word_vectors = TEXT.vocab.vectors[text].to(device)
                    label = label.to(device)
                    hidden = model.init_hidden(word_vectors.size(1)).to(device)

                    output, _ = model(word_vectors, hidden)
                    _, predicted = torch.max(output, 1)  # 获取每个样本的预测类别，第二个输出为最大值索引

                    total_samples += label.size(0)
                    total_correct += (predicted == label).sum().item()  # 计算预测正确的样本数量

            accuracy = total_correct / total_samples  # 计算准确率
            writer.add_scalar('Accuracy/train', accuracy, epoch)

            total_correct = 0
            total_samples = 0

            with torch.no_grad():  # 在验证集或测试集上不进行梯度计算
                for batch in valid_iterator:  # 使用测试迭代器（或验证迭代器）进行迭代
                    text, label = batch.review, batch.cat
                    word_vectors = TEXT.vocab.vectors[text].to(device)
                    label = label.to(device)
                    hidden = model.init_hidden(word_vectors.size(1)).to(device)

                    output, _ = model(word_vectors, hidden)
                    _, predicted = torch.max(output, 1)  # 获取每个样本的预测类别

                    total_samples += label.size(0)
                    total_correct += (predicted == label).sum().item()  # 计算预测正确的样本数量

            accuracy = total_correct / total_samples  # 计算准确率
            writer.add_scalar('Accuracy/validation', accuracy, epoch)
        
    # 混淆矩阵
    total_conf_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():  # 在验证集或测试集上不进行梯度计算
        for batch in test_iterator:  # 使用测试迭代器（或验证迭代器）进行迭代
            text, label = batch.review, batch.cat
            word_vectors = TEXT.vocab.vectors[text].to(device)
            label = label.to(device)
            hidden = model.init_hidden(word_vectors.size(1)).to(device)

            output, _ = model(word_vectors, hidden)
            _, predicted = torch.max(output, 1)  # 获取每个样本的预测类别
   
            # 计算当前batch混淆矩阵
            cm = confusion_matrix(label.cpu(), predicted.cpu(), labels=range(10))

            total_conf_matrix += cm

        # 计算指标
        true_positives = np.diag(total_conf_matrix)
        false_positives = np.sum(total_conf_matrix, axis=0) - true_positives
        false_negatives = np.sum(total_conf_matrix, axis=1) - true_positives
        total_samples = np.sum(total_conf_matrix)

        # 计算准确率
        accuracy = true_positives.sum() / total_samples

        # 计算召回率
        recall = true_positives / (true_positives + false_negatives)
        recall = np.mean(recall)

        # 计算精确率
        precision = true_positives / (true_positives + false_positives)
        non_zero_denominators = (true_positives + false_positives) != 0
        precision_non_zero = precision[non_zero_denominators]
        precision = np.mean(precision_non_zero)

        # 计算 F1 值
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 打印准确率、召回率和 F1 值
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"F1 Score: {f1_score}")

        plot_matrix(log_dir + '/matrix.png', range(10), total_conf_matrix, 'confusion_matrix', axis_labels=['书籍', '平板', '手机', '水果', '洗发水', '热水器', '蒙牛', '衣服', '计算机', '酒店'])

        torch.save(model, log_dir + f'/model_{accuracy}_{f1_score}.pth')

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM', help='')
    parser.add_argument('--lr', type=float, default=0.001, help='learing rate')
    args = parser.parse_args()
    main(**vars(args))