## 基于统计学方法的异常检测
### 1、概述

#### 1.1 基本原理
首先假设正常数据服从某一分布，对于异常点，并不符合该分布，由此可以利用统计学方法找到异常点。一般思想为：学习一个拟合给定数据集的生成模型，然后识别该模型低概率区域中的对象，把它们作为异常点。

#### 1.2 主要类型
- 参数方法
假定正常的数据对象被一个以$\Theta$为参数的参数分布产生。该参数分布的概率密度函数$f(x,\Theta)$给出对象$x$在该分布下的概率。该值越小，$x$越可能是异常点。
- 非参数方法
并不假定先验分布，而是通过输入数据确定模型，并不是完全无参，只是参数的个数和性质很灵活。

### 2、参数方法
#### 2.1 正态分布下的异常点检测
对于一维数据集$\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$，可以假定它们服从一元正态分布：$x^{(i)}\sim N(\mu, \sigma^2)$，对于参数$\mu$和$\sigma$，可以用矩法估计的思想求出来：

$\mu=\frac 1m\sum_{i=1}^m x^{(i)}$

$\sigma^2=\frac 1m\sum_{i=1}^m (x^{(i)}-\mu)^2$

得到参数后就意味着找到了正态分布的概率密度函数；接下来可以设定阈值，以此找到异常点（概率小于该阈值的数据）例如：常见的3$\sigma$原则：超出$(\mu-3\sigma, \mu+3\sigma)$的数据可以被认为是异常点；也可以利用箱线图，找到上下四分位数（		Q1、Q3），此时，异常点常被定义为小于Q1－1.5IQR或大于Q3+1.5IQR的那些数据。
利用如下的Python代码可以很容易的画出箱线图

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.random.randn(50000) * 20 + 20
sns.boxplot(data=data)
```
结果如下

<img src="E:\datawhale\AnomalyDetection\AnomalyDetection\img\箱线图.png" style="zoom:150%;" />

#### 2.2 多元数据下的异常点检测
当数据的维度大于1时，可以将一元异常点的检测方法推广到多元，具体方法为：多元异常点检测任务转换成一元异常点检测问题。例如基于正态分布的一元异常点检测扩充到多元情形时，可以求出每一维度的均值和标准差（利用矩法估计），此时的概率密度函数为：
$p(x)=\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^n\frac 1{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})$
上式适用于各个维度相互独立的情况，在各个维度之间由关联时，需要用到多元高斯分布

- 多元高斯分布
$\mu=\frac{1}{m}\sum^m_{i=1}x^{(i)}$
$\sum=\frac{1}{m}\sum^m_{i=1}(x^{(i)}-\mu)(x^{(i)}-\mu)^T$
$p(x)=\frac{1}{(2 \pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)$

### 3 非参数方法
对数据集进行先验分布假定是一件主观性很强的事，而非参数方法不受参数的拘束，因此在很多情况下都适用。
- 直方图法
1.构造直方图。使用输入数据（训练数据）构造一个直方图。该直方图可以是一元的，或者多元的（如果输入数据是多维的）。指定直方图的类型（等宽的或等深的）和其他参数（直方图中的箱数或每个箱的大小等）。与参数方法不同，这些参数并不指定数据分布的类型。
2.检测异常点。为了确定一个对象是否是异常点，可以对照直方图检查它。在最简单的方法中，如果该对象落入直方图的一个箱中，则该对象被看作正常的，否则被认为是异常点。也可以使用直方图赋予每个对象一个异常点得分。例如令对象的异常点得分为该对象落入的箱的容积的倒数。

使用直方图检验异常点的方法也有缺陷：对箱尺的要求严格，尺寸太小，正常点落不到箱中，尺寸太大，部分异常点成为漏网之鱼

### 4、HBOS（Histogram-based Outlier Score）
1.为每个数据维度做出数据直方图。对分类数据统计每个值的频数并计算相对频率。对数值数据根据分布的不同采用以下两种方法：

* 静态宽度直方图：标准的直方图构建方法，在值范围内使用k个等宽箱。样本落入每个桶的频率（相对数量）作为密度（箱子高度）的估计。时间复杂度：$O(n)$

* 动态宽度直方图：首先对所有值进行排序，然后固定数量的$\frac{N}{k}$个连续值装进一个箱里，其	中N是总实例数，k是箱个数；直方图中的箱面积表示实例数。因为箱的宽度是由箱中第一个值和最后一个值决定的，所有箱的面积都一样（因为箱子中的数量一样），因此每一个箱的高度都是可计算的。这意味着跨度大的箱的高度低，即密度小，只有一种情况例外，超过k个数相等，此时允许在同一个箱里超过$\frac{N}{k}$值。

  时间复杂度：$O(n\times log(n))$

2.对每个维度都计算了一个独立的直方图，其中每个箱子的高度表示密度的估计。然后为了使得最大高度为1（确保了每个特征与异常值得分的权重相等），对直方图进行归一化处理。最后，每一个实例的HBOS值由以下公式计算：
$$
H B O S(p)=\sum_{i=0}^{d} \log \left(\frac{1}{\text {hist}_{i}(p)}\right)
$$
推导如下:

假设样本*p*第 *i* 个特征的概率密度为$p_i(p)$ ，则*p*的概率密度可以计算为：
$$
P(p)=P_{1}(p) P_{2}(p) \cdots P_{d}(p)
$$
两边取对数：
$$
\begin{aligned}
\log (P(p)) &=\log \left(P_{1}(p) P_{2}(p) \cdots P_{d}(p)\right) =\sum_{i=1}^{d} \log \left(P_{i}(p)\right)
\end{aligned}
$$
概率密度越大，异常评分越小，为了方便评分，两边乘以“-1”：
$$
-\log (P(p))=-1 \sum_{i=1}^{d} \log \left(P_{t}(p)\right)=\sum_{i=1}^{d} \frac{1}{\log \left(P_{i}(p)\right)}
$$
最后可得：
$$
H B O S(p)=-\log (P(p))=\sum_{i=1}^{d} \frac{1}{\log \left(P_{i}(p)\right)}
$$
HBOS在全局异常检测问题上表现良好，但不能检测局部异常值。但是HBOS比标准算法快得多，尤其是在大数据集上。

### 4 利用pyod库调用HBOS

```python
from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.hbos import HBOS
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, y_train, X_test, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # train HBOS detector
    clf_name = 'HBOS'
    clf = HBOS()
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # visualize the results
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
```

结果及可视化如下：

![roc](E:\datawhale\AnomalyDetection\AnomalyDetection\img\hbos-roc.png)

<img src="E:\datawhale\AnomalyDetection\AnomalyDetection\img\hbos-可视化.png"  />

HBOS算法在训练集上的ROC为0.8，在验证集上的ROC为0.6，效果一般