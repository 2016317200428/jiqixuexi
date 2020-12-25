#### 实验环境
python3.5

实验数据

有两个.csv文件，里面是将图片处理好的数据，rgbhsv.scv文件是rgbhsv六个特征值，r_g_b.csv是两两特征分量的数据，由于实验图片为实验室资源不能上传

数据预处理

load_data_into_csv.py是预处理红提图片的数据文件，可以得到六个特征分量，load_r_g_b_features.py得到的是rgb两两特征值之间的比值

线性模型

mutil_LinearRegression.py是多元线性回归模型代码，
single_linear_regression.py是一元线性回归模型
