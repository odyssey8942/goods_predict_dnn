#coding: UTF-8
import math
import random
import pandas as pd
import datetime as dt
import numpy as np
import cupy as cp
import scipy.stats
import matplotlib.pylab as plt
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, optimizer, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import csv

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1=L.Linear(7,14),
            l2=L.Linear(14,10),
            l3=L.Linear(10,6),
            l4=L.Linear(6,4),
            l5=L.Linear(4,2),
            l6=L.Linear(2,1),
        )
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = F.relu(self.l5(h))
        return self.l6(h)
    


    #正規化
def get_data(x,t):
    #教師データ
    train_x, train_t = [], []
    train_path = "train.csv"
    test_path = "test.csv"    
    csv_file = open(train_path, "r", encoding="utf_8", errors="", newline="\n" )
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    
    #教師データ変換
    for row in f:
        """
        obj = row[2:8]
        del obj[3]
        obj[1] = float(obj[1])*float(10.0/36.0)
        train_x.append(obj)
        """
        obj = row[2:12]

        if(obj[0] == '月'):
            obj[0] = 0
        elif(obj[0] == '火'):
            obj[0] = 1
        elif(obj[0] == '水'):
            obj[0] = 2
        elif(obj[0] == '木'):
            obj[0] = 3
        elif(obj[0] == '金'):
            obj[0] = 4
        else:
            pass
        
        if(obj[3] == ''):
            obj[3] = '400'
        
        if(obj[6] == ''):
            obj[6] = '0'

        if(obj[7] == '快晴'):
            obj[7] = 0
        elif(obj[7] == '晴れ'):
            obj[7] = 1
        elif(obj[7] == '薄曇'):
            obj[7] = 2
        elif(obj[7] == '曇'):
            obj[7] = 3
        elif(obj[7] == '雨'):
            obj[7] = 4
        elif(obj[7] == '雷電'):
            obj[7] = 5
        elif(obj[7] == '雪'):
            obj[7] = 6
        else:
            pass

        if(obj[8] == '--'):
            obj[8] = '0'
        
        
        del obj[4:6]
        del obj[2]

        train_x.append(obj)
        train_t.append(row[1])

    del train_x[0]
    del train_t[0]
    for row in train_x:
        print(row)
    train_x = np.array(train_x, dtype="float32")
    train_t = np.array(train_t, dtype="float32")

    return scipy.stats.zscore(train_x),train_t
epoch = 10

x,t = [],[]
x,t = get_data(x,t)
num = len(x)

x = Variable(x)
t = Variable(t)

model = Model()
optimizer = optimizers.Adam()
optimizer.setup(model)

#while(1):
for i in range(2000):
    #for j in range(num):
    model.cleargrads()
    y = model(x)
    #print(y.data)
    loss = F.mean_squared_error(y,t.reshape(num,1))
    loss.backward()
    optimizer.update()
    print("loss:",loss.data)

test_path = "test.csv"    
csv_file = open(test_path, "r", encoding="utf_8", errors="", newline="\n" )
test_f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)


test_x = []
#テストデータ変換
for row in test_f:
    """
    obj = row[2:8]
    del obj[3]
    obj[1] = float(obj[1])*float(10.0/36.0)
    train_x.append(obj)
    """
    obj = row[1:12]

    if(obj[0] == '月'):
        obj[0] = 0
    elif(obj[0] == '火'):
        obj[0] = 1
    elif(obj[0] == '水'):
        obj[0] = 2
    elif(obj[0] == '木'):
        obj[0] = 3
    elif(obj[0] == '金'):
        obj[0] = 4
    else:
        pass
    
    if(obj[3] == ''):
        obj[3] = '400'
    
    if(obj[6] == ''):
        obj[6] = '0'

    if(obj[7] == '快晴'):
        obj[7] = 0
    elif(obj[7] == '晴れ'):
        obj[7] = 1
    elif(obj[7] == '薄曇'):
        obj[7] = 2
    elif(obj[7] == '曇'):
        obj[7] = 3
    elif(obj[7] == '雨'):
        obj[7] = 4
    elif(obj[7] == '雷電'):
        obj[7] = 5
    elif(obj[7] == '雪'):
        obj[7] = 6
    else:
        pass

    if(obj[8] == '--'):
        obj[8] = '0'
    
    
    del obj[4:6]
    del obj[2]

    test_x.append(obj)

del test_x[0]
test_x = np.array(test_x, dtype="float32")
test_x = scipy.stats.zscore(test_x)
test_x = Variable(test_x)
y = model(test_x)

print(y.data)
serializers.save_npz("lawson_dnn.npz",model)



     

