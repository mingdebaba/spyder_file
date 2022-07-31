from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib as np
import matplotlib.pyplot as plt

df = pd.read_csv('kion10y.csv', encoding="utf-8")
train_year = (df["年"] <=2018)
test_year = (df["年"]>=2019)
interval = 6

def make_data(data):
    x=[]
    y=[]
    temps= list(data["気温"])
    for i in range(len(temps)):
        if i< interval:continue
        y.append(temps[i])
        xa=[]
        for p in range(interval):
            d=i+p-interval
            xa.append(temps[d])
        x.append(xa)
    return (x,y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

# 直線回帰分析を行う ---(*3)
lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y) # 学習
pre_y = lr.predict(test_x) # 予測



pre_y - test_y
# 結果を図にプロット ---(*4)
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()

