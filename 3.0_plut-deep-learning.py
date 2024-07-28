import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

print("Parameters")
d = 2  # dimension, need modify to higher dimension
n = 50  # Population
m = 20  # Best sites
e = 8  # elite sites
nep = 50  # Elite bees
nsp = 25  # other bees
ngh = 0.1  # local search radius

MaxIt = 200  # iteration
sg = 100  # study_gap

# Aim function (Sphere Function)
def Sphere(x):
    return np.sum(np.power(x, 2))

def Cost(x):
    return Sphere(x)

def Foraging(x, ngh):
    nVar = x.size
    k = np.random.randint(0, nVar)
    y = x.copy()
    y[0, k] = x[0, k] + np.random.uniform(-ngh, ngh)
    return y

# Init each bee's position and cost 
class Bee:
    def __init__(self, Position, Cost):
        self.Position = Position
        self.Cost = Cost

# 设置蜜蜂算法参数/ Setting bees algorithm parametric
varMin = float(-10)
varMax = float(10)
maxIt = MaxIt
nScoutBee = n
nSelectedSite = m
nEliteSite = e
nSelectedSiteBee = nsp
nEliteSiteBee = nep
shrink = 0.95

# 初始化/initialization
bee = []
for i in range(nScoutBee):
    position = np.random.uniform(varMin, varMax, size=(1, d))
    bee.append(Bee(position, Cost(position)))

bee.sort(key=lambda bee: bee.Cost)

BestSol = bee[0]
BestCost = np.zeros([maxIt, 1])
BestPos = []

history_positions = []
history_fitnesses = []

# 创建深度学习模型/Creat a deep learning Model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# 训练模型/ training model
def train_model(history_positions, history_fitnesses, epochs=20):
    model = create_model(input_dim=history_positions.shape[1])
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    model.fit(history_positions, history_fitnesses, epochs=epochs, batch_size=32, verbose=0, callbacks=[lr_scheduler])
    return model

# 预测新的位置/ Predicting the new position
def predict_new_position(model, current_position):
    new_position = current_position.copy()
    perturbation = model.predict(np.array([current_position]))[0]
    new_position += perturbation
    return new_position

# 主循环/ Main loop
model = None
for it in range(maxIt):
    # 自动数据收集/ Automated data collection
    for bee_obj in bee:
        history_positions.append(bee_obj.Position.flatten())
        history_fitnesses.append(bee_obj.Cost)

    # 确保所有适应度数据为标量
    history_fitnesses = [float(fitness) for fitness in history_fitnesses]

    # 打印调试信息 / Print debug information
    print(f"Iteration {it}: history_positions length = {len(history_positions)}, study gap = {sg}")

    # 动态训练模型/ dynamic training model
    if len(history_positions) > sg and it % sg == 0:
        print(f"Training model at iteration {it}")
        history_positions_np = np.array(history_positions)
        history_fitnesses_np = np.array(history_fitnesses)

        # 确保数据转换前的一致性/ Ensure consistency before data conversion
        if len(history_positions_np.shape) == 2 and len(history_fitnesses_np.shape) == 1:
            model = train_model(history_positions_np, history_fitnesses_np, epochs=20)
            print(f"Model trained at iteration {it}")

    # 精英位置/ Elite position
    for i in range(nEliteSite):
        bestnewbee = Bee([], math.inf)
        for j in range(nEliteSiteBee):
            if model is not None:
                new_position = predict_new_position(model, bee[i].Position.flatten()).reshape(1, -1)
                new_fitness = Cost(new_position)
            else:
                new_position = Foraging(bee[i].Position, ngh)
                new_fitness = Cost(new_position)
            if new_fitness < bestnewbee.Cost:
                bestnewbee = Bee(new_position, new_fitness)
        if bestnewbee.Cost < bee[i].Cost:
            bee[i] = bestnewbee

    # 非精英位置/ Non-Elite position
    for i in range(nEliteSite, nSelectedSite):
        bestnewbee = Bee([], math.inf)
        for j in range(nSelectedSiteBee):
            new_position = Foraging(bee[i].Position, ngh)
            new_fitness = Cost(new_position)
            if new_fitness < bestnewbee.Cost:
                bestnewbee = Bee(new_position, new_fitness)
        if bestnewbee.Cost < bee[i].Cost:
            bee[i] = bestnewbee

    # 非选择位置/ Non-Selected sites
    for i in range(nSelectedSite, nScoutBee):
        position = np.random.uniform(varMin, varMax, size=(1, d))
        bee[i] = Bee(position, Cost(position))

    # 排序 / sort
    bee.sort(key=lambda bee: bee.Cost)

    # 更新最优解 / Update
    BestSol = bee[0]

    # 存储最优成本 / Store Best Cost ever found
    BestCost[it] = BestSol.Cost
    BestPos.append(BestSol.Position)

    # 显示迭代信息 / Print iteration info
    print(f"Iteration {it}: Best Cost = {BestCost[it][0]}, Best Position = {BestPos[it].flatten()}")

    ngh = shrink * ngh

# 结果展示 / print the results
Y = BestCost
X = list(range(len(Y)))

plt.grid(True, which="both")
plt.semilogy(X, Y)
plt.ylim([0, 10])
plt.xlim([0, len(X)])
plt.title('Sphere Function')
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.show()
