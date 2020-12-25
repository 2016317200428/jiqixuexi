import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def h(x):
    return w0 + w1 * x


if __name__ == '__main__':
    # alpha学习率
    rate = 0.000001

    # y = w0 * x + w1
   # w0 = np.random.normal()
   # w1 = np.random.normal()
    w0 = 25.662270
    w1 = -0.074749



    adv_data = pd.read_csv("F:/sugar_new/images_data/CSV/rgbhsv.csv")
    new_adv_data = adv_data.iloc[:, 0:]
  #  x_train, X_test, y_train, Y_test = train_test_split(new_adv_data.iloc[:, 2], new_adv_data.label, train_size=0.8)
    x_train = new_adv_data.iloc[:, 3]
    y_train = new_adv_data.label
    # 训练数据
   # x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
   # y_train = np.array([1, 3, 4, 5, 6, 7, 8, 9, 10])
    err = 1
    # 计算误差函数
    max_iter = 8000
    for i in range(max_iter):
        for (x, y) in zip(x_train, y_train):
            w0 -= (rate * (h(x) - y) * 1)
            w1 -= (rate * (h(x) - y) * x)

        # 代入找误差
        err = 0.0
        for (x, y) in zip(x_train, y_train):
            err += (y - h(x)) ** 2
        err /= float(x_train.size * 2)
        if i%2000==0 :
            print(err)

    # 打印
    print("w0的值为%f" % w0)
    print("w1的值为%f" % w1)
    print("误差率的值为%f" % err)


    print(y_train)
    print(x_train)
    p = np.corrcoef(y_train, x_train)
    print("pearsonr: ",p)

    fig = plt.figure(figsize=(5, 5), dpi=160)
    ax1 = fig.add_subplot(1, 1, 1)
    # 使用scatter方法绘制散点图，和之前绘制折线图图的唯一区别
    plt.scatter(x_train, y_train, label="样本点")
    plt.plot(x_train, h(x_train), 'k', label='拟合曲线')
    # 添加图例
    plt.legend(loc="upper left")
    # 添加描述信息
    plt.xlabel("H")
    plt.ylabel("糖度")
    plt.title("Pearson="+str(p[1][0]))
    ax1.legend(loc='best')
    plt.show()