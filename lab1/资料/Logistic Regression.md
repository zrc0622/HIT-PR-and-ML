# Logistic 回归

模型$h_\theta(x)=g(\theta^\prime x)=\frac{1}{1+exp(-\theta^\prime x)}$

训练集: $\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$

其中，$x=(x_0,x_1,...,x_n)'_{n+1}\in R^{n+1},x_0=1$,

$y\in \{0,1\}$,

$\theta=(\theta_0,\theta_1,...,\theta_n)'_{n+1}\in R^{n+1}$,

$g(\cdot)=\frac{1}{1+exp(-\cdot)}$称为sigmoid函数

损失函数: $J(\theta)=\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$

$\nabla J(\theta)=\frac{1}{m}X'(g(X\theta)-Y)$，

其中，

$X=\begin{pmatrix}x^{(1)'}\\x^{(2)'}\\\vdots\\x^{(m)'}\end{pmatrix}=\begin{pmatrix}1&x^{(1)'}_1&\cdots&x^{(1)'}_n\\1&x^{(2)'}_1&\cdots&x^{(2)'}_n\\\vdots&\vdots&\ddots&\vdots\\1&x^{(m)'}_1&\cdots&x^{(m)'}_n\end{pmatrix}\in R^{m\times(n+1)}$称为design matrix

## 特征规范化 

设$x^{(i)}_j$为第$i$个训练样本的第$j$维的特征，

$x_i\gets \frac{x_i-\mu_i}{s_i}$

其中，$\mu_i=\frac{1}{m}\sum^{m}_{j=1}x_{i}^{j}$, $s_i$为$max\{x_i\}-min\{x_i\}$或$std(x_i)$

分子和分母的部分分别叫做Mean Normalization和Feature Scaling。

## 梯度下降

学习率的选取

## 非线性二分类

构造高维特征: 原特征$x\in R$，可以构造特征$(1,x,x^2)$，$(1,x,x^2,x^3)$或$(1,x,\sqrt{x})$等等。决策边界(decision boundary)也就由直线/平面变为了曲线/曲面。

以线性回归为例，原模型$h_\theta(x)=\theta_0+\theta_1 x$变为$h_\theta(x)=\theta_0+\theta_1 x+\theta_2x^2$，

$h_\theta(x)=\theta_0+\theta_1 x+\theta_2x^2+\theta_3x^3$

或$h_\theta(x)=\theta_0+\theta_1 x+\theta_2\sqrt{x}$

## 处理过拟合

- 减少参数量(减少特征数量)
- 使用正则化(Regularization)

加入$L_2$正则化的损失函数:  $J(\theta)=\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$

