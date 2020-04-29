import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import tree
from sklearn.model_selection import cross_val_score

data = pd.read_csv('./dataset/sonar.all-data',header=None,sep=',')
print(data.head())
X = data.iloc[:,:-1]
y = data.iloc[:,-1:].values.flatten()
chrom_length = len(data.columns)-1    # 染色体长度

#############################################
# 影响迭代次数
tmp = math.exp(3)   # 初始温度
tmp_min = math.exp(-8)    # 停止温度
alpha = 0.98    # 降温系数

counter = 10000     # 生成更差解的次数

# 影响较差解采纳概率,越大越容易采纳更差解
k = 0.002
#############################################

# 是否可采纳
def is_acceptable(delta_E,tmp):     
    if delta_E<=0:   # ΔE<=0，直接采纳
        print('直接采纳')
        return True

    p=math.exp(-delta_E/(k*tmp))    # 求采纳概率
    r=random.random()
    if r<p:
        print(str(r)+"<"+str(p)+"，可采纳") 
        return True
    else:
        print(str(r)+">="+str(p)+"，不采纳"+"（"+str(counter)+"）")
        return False

# 特征编码
def geneEncoding():
    while True:
        temp = []
        has_1 = False   # 这条染色体是否有1
        for j in range(chrom_length):
            rand = random.randint(0,1)
            if rand == 1:
                has_1 = True
            temp.append(rand)
        if has_1:   # 染色体不能全0
            return temp

# 获取适应度
def getFitness(x):
    X_test = X

    has_1 = False
    for j in range(chrom_length):
        if x[j] == 0:
            X_test =X_test.drop(columns = j)
        else:
            has_1 = True
    X_test = X_test.values
        
    if has_1:
        clf = tree.DecisionTreeClassifier() # 决策树作为分类器
        fitness = cross_val_score(clf, X_test, y, cv=5).mean()  # 5次交叉验证
        return fitness
    else:
        fitness = 0     # 全0的适应度为0
        return fitness

# 从旧解生成新解
def getNewChrom(x):
    mpoint = random.randint(0, chrom_length-1)  # 随机选择变异点
    if x[mpoint] == 1:
        x[mpoint] = 0
    else:
        x[mpoint] = 1
    return x

# 程序入口
if __name__=='__main__':
    plt.xlabel('temperature')
    plt.ylabel('fitness')
    plt.xlim((tmp_min,tmp))    # x坐标范围
    plt.ylim((0.4,0.9)) # y坐标范围
    px = []
    py_old = []
    py_new = []
    plt.ion()
    
    x_old = geneEncoding()    # 生成初始随机解
    E_old = getFitness(x_old)

    while tmp > tmp_min:

        x_new = getNewChrom(x_old)   # 生成随机解
        E_new = getFitness(x_new)
        delta_E = -(E_new - E_old)

        if is_acceptable(delta_E,tmp):  # 可采纳
            x_old = x_new
            E_old = E_new

        if delta_E<=0:   # ΔE<=0，降温
            tmp = tmp * alpha
        else:
            counter -= 1

        if counter < 0:
            break

        print(tmp)
        print(x_old)
        print(E_old)
        print()

        px.append(tmp)  # 画图
        py_old.append(E_old)
        py_new.append(E_new)
        plt.plot(px,py_old,'r')
        plt.plot(px,py_new)
        plt.show()
        plt.pause(0.001)
    
