import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stats
import sys
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
from sklearn.linear_model import LinearRegression
import numpy as np
from dataset import load_boston

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

adv_data = pd.read_csv("F:/sugar_new/images_data/CSV/rgbhsv.csv")
new_adv_data = adv_data.iloc[:, 0:]


#data=load_boston()
#X=data.data
#Y=data.target

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.05)
X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.iloc[:, :5], new_adv_data.label, train_size=0.80)
# 为便于numpy矩阵乘法之间的维度维护，统一将Y转换成列向量，与一行一个样本对应
Y_train=Y_train.values.reshape((-1,1))
Y_test=Y_test.values.reshape((-1,1))
#print(X_train)


def linear_reg(X, Y, alpha=0.000001, max_iter=80000):
    Y = Y.reshape((-1, 1))

    n_sample = X.shape[0]
    n_feature = X.shape[1]

    W = np.random.randn(n_feature).reshape((n_feature, 1))  # 权重
    b = 1  # 偏置

    for i in range(max_iter):
        Y_hat = np.dot(X, W) + b

        dW = 2 * X.T.dot(Y_hat - Y) / n_sample
        db = 2 * np.sum(Y_hat - Y) / n_sample

        W = W - alpha * dW
        b = b - alpha * db

        if i % 2000 == 0:
            Y_hat = np.dot(X, W) + b
            L = np.sum((Y - Y_hat) ** 2) ** 0.5 / n_sample
            print(L)

    return W, b


W, b = linear_reg(X_train, Y_train)
print("W:",W)
print("b:",b)

Y_valid_predict = np.dot(X_train, W) + b
Y_test_predict = np.dot(X_test, W) + b
Y_valid_predict = Y_valid_predict.reshape(1,len(Y_valid_predict))
Y_train = Y_train.reshape(1,len(Y_train))
Y_test = Y_test.reshape(1, len(Y_test))
Y_test_predict = Y_test_predict.reshape(1, len(Y_test_predict))

print("验证集相关系数：", np.corrcoef(Y_valid_predict, Y_train))
print("测试集相关系数：", np.corrcoef(Y_test_predict, Y_test))

plt.figure()
plt.plot(range(len(Y_test_predict)),Y_test_predict, 'b', label="predict")
plt.plot(range(len(Y_test_predict)), Y_test, 'r', label="test")
plt.legend(loc="upper right")  # 显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
#plt.savefig("ROC.jpg")
plt.show()


#pred = pd.Series(Y_valid_predict)
#label = pd.Series(Y_train)
g = (sns.jointplot(Y_train, Y_valid_predict, kind = "reg").set_axis_labels("gronndTruth", "prediction"))
g.ax_marg_x.set_xlim(13,18)
g.ax_marg_y.set_ylim(13,18)
g.annotate(stats.pearsonr)
plt.show()
