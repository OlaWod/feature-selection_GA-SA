import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score

data = pd.read_csv('./dataset/sonar.all-data',header=None,sep=',')
print(data.head())
X = data.iloc[:,:-1]
y = data.iloc[:,-1:].values.flatten()

clf = tree.DecisionTreeClassifier() # 决策树作为分类器
fitness = cross_val_score(clf, X, y, cv=5).mean()  # 5次交叉验证
print(fitness)
