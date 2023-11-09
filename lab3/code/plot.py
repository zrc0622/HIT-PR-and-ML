import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# plot_matrix函数参数包括：
# y_true样本的真实标签，为一向量
# y_pred样本的预测标签，为一向量，与真实标签长度相等
# labels_name样本在数据集中的标签名，如在示例中，样本的标签用0, 1, 2表示，则此处应为[0, 1, 2]
# title=None图片的标题
# thresh=0.8临界值，大于此值则图片上相应位置百分比为白色
# axis_labels=None最终图片中显示的标签名，如在示例中，样本标签用0, 1, 2表示分别表示失稳、稳定与潮流不收敛，我们最终图片中显示后者而非前者，则可令此参数为[‘unstable’, ‘stable’, ‘non-convergence’]

def plot_matrix(dir, labels_name, cm, title=None, thresh=0.8, axis_labels=None):
    plt.figure(figsize=(8, 7))
    #  利用sklearn中的函数生成混淆矩阵并归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=plt.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    plt.savefig(dir,  dpi=300)
