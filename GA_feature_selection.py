import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score

data = pd.read_csv('./dataset/sonar.all-data',header=None,sep=',')
print(data.head())
X = data.iloc[:,:-1]
y = data.iloc[:,-1:].values.flatten()

iterations = 100 # 迭代次数
pop_size = 100   # 种群大小，多少个染色体
pc = 0.25   # 交叉概率
pm = 0.01   # 变异概率

chrom_length = len(data.columns)-1    # 染色体长度
pop = []    # 种群
fitness_list = []   # 适应度
ratio_list = []     # 累计概率


# 初始化种群
def geneEncoding():
    i = 0
    while i < pop_size:
        temp = []
        has_1 = False   # 这条染色体是否有1
        for j in range(chrom_length):
            rand = random.randint(0,1)
            if rand == 1:
                has_1 = True
            temp.append(rand)
        if has_1:   # 染色体不能全0
            i += 1
            pop.append(temp)
        

# 计算适应度
def calFitness():
    fitness_list.clear()
    for i in range(pop_size):   # 计算种群中每条染色体的适应度
        X_test = X

        has_1 = False
        for j in range(chrom_length):
            if pop[i][j] == 0:
                X_test =X_test.drop(columns = j)
            else:
                has_1 = True
        X_test = X_test.values
        
        if has_1:
            clf = tree.DecisionTreeClassifier() # 决策树作为分类器
            fitness = cross_val_score(clf, X_test, y, cv=5).mean()  # 5次交叉验证
            fitness_list.append(fitness)
        else:
            fitness = 0     # 全0的适应度为0
            fitness_list.append(fitness)

# 计算适应度的总和
def sumFitness():
    total = 0
    for i in range(pop_size):
        total += fitness_list[i]
    return total

# 计算每条染色体的累计概率
def getRatio():
    ratio_list.clear()
    ratio_list.append(fitness_list[0])
    for i in range(1, pop_size):
        ratio_list.append(ratio_list[i-1] + fitness_list[i])
    ratio_list[-1] = 1

# 选择
def selection():
    global pop
    total_fitness = sumFitness()
    for i in range(pop_size):
        fitness_list[i] = fitness_list[i] / total_fitness
    getRatio()
    
    rand_ratio = [] # 随机概率
    for i in range(pop_size):
        rand_ratio.append(random.random())
    rand_ratio.sort()

    new_pop = []    # 新种群
    i = 0  # 已经处理的随机概率数
    j = 0  # 超出范围的染色体数
   
    while i < pop_size:
        if rand_ratio[i] < ratio_list[j]:   # 随机数在第j个染色体的概率范围内
            new_pop.append(pop[j])
            i += 1
        else:
            j += 1

    pop = new_pop

# 交叉
def crossover():
    for i in range(pop_size-1): # 若交叉，则染色体i与染色体i+1交叉
        if random.random() < pc:# 发生交叉
            cpoint = random.randint(0, chrom_length-1)    # 随机选择交叉点
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][:cpoint])
            temp1.extend(pop[i+1][cpoint:])
            temp2.extend(pop[i+1][:cpoint])
            temp2.extend(pop[i][cpoint:])
            pop[i] = temp1
            pop[i+1] = temp2

# 变异
def mutation():
    for i in range(pop_size):
        if random.random() < pm: # 发生变异
            mpoint = random.randint(0, chrom_length-1)  # 随机选择变异点
            if pop[i][mpoint] == 1:
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1

# 最优解
def getBest():
    best_chrom = pop[0]
    best_fitness = fitness_list[0]
    for i in range(1,pop_size):
        if fitness_list[i] > best_fitness:
            best_fitness = fitness_list[i]  # 最佳适应值
            best_chrom = pop[i] # 最佳染色体

    return best_chrom, best_fitness

if __name__=='__main__':
    
    plt.xlabel('iterations')
    plt.ylabel('best fitness')
    plt.xlim((0,iterations))    # x坐标范围
    plt.ylim((0,1)) # y坐标范围
    px = []
    py = []
    plt.ion()
    
    results = []
    geneEncoding() # 初始化种群
    for i in range(iterations):
        print(i)
        
        calFitness() # 计算种群中每条染色体适应度
        
        best_chrom, best_fitness = getBest()
        results.append([i, best_chrom, best_fitness])
        
        selection() # 选择
        crossover() # 交叉
        mutation()  # 变异

        print([i, best_chrom, best_fitness])
        
        px.append(i)    # 画图
        py.append(best_fitness)
        plt.plot(px,py)
        plt.show()
        plt.pause(0.001)
        

