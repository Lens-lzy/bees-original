import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plot

# 环境参数
d = 2  # 维度
n = 50  # 种群数量
m = 20  # 最佳位置数量
e = 8  # 精英位置数量
nep = 50  # 精英位置的蜜蜂数量
nsp = 25  # 选择位置的蜜蜂数量
ngh = 0.1  # 局部搜索半径
MaxIt = 500  # 最大迭代次数
varMin = -10
varMax = 10

# 定义目标函数
def Sphere(x):
    return np.sum(np.power(x, 2))

def Cost(x):
    return Sphere(x)

# 改进的演员-评论员模型
class ImprovedActorCritic(tf.keras.Model):
    def __init__(self, action_space):
        super(ImprovedActorCritic, self).__init__()
        self.common = layers.Dense(256, activation="relu")  # 增加层数和神经元数量
        self.actor = layers.Dense(action_space, activation="tanh")
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

# 初始化改进的演员-评论员模型
num_actions = d - 1
model = ImprovedActorCritic(num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 使用自适应学习率优化器
huber_loss = tf.keras.losses.Huber()

def train_step(state, action, reward, next_state):
    with tf.GradientTape() as tape:
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        action_probs, critic_value = model(state)
        _, next_critic_value = model(next_state)

        target = reward + 0.99 * next_critic_value
        advantage = target - critic_value

        actor_loss = -tf.reduce_sum(tf.math.log(action_probs) * advantage)
        critic_loss = huber_loss(tf.expand_dims(target, 0), tf.expand_dims(critic_value, 0))

        loss = actor_loss + critic_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 初始化蜜蜂种群
class Bee:
    def __init__(self, Position, Cost):
        self.Position = Position
        self.Cost = Cost

bee = []
for i in range(n):
    position = np.random.uniform(varMin, varMax, size=(1, d-1))
    bee.append(Bee(position, Cost(position)))

bee.sort(key=lambda bee: bee.Cost, reverse=False)
BestSol = bee[0]
BestCost = np.zeros([MaxIt, 1])
BestPos = [None] * MaxIt

# 主循环
for it in range(MaxIt):
    # 更新精英蜜蜂位置
    for i in range(e):
        state = bee[i].Position
        for _ in range(nep):
            action, _ = model(state)
            action = action.numpy() + np.random.normal(scale=0.1, size=action.shape)  # 随机噪声注入
            new_position = state + action * ngh
            new_position = np.clip(new_position, varMin, varMax)
            new_cost = Cost(new_position)
            reward = bee[i].Cost - new_cost - 0.1 * np.linalg.norm(action)  # 引入惩罚项
            train_step(state, action, reward, new_position)
            state = new_position

            if new_cost < bee[i].Cost:
                bee[i] = Bee(new_position, new_cost)

    # 更新选择位置的蜜蜂
    for i in range(e, m):
        bestnewbee = Bee([], np.inf)
        for _ in range(nsp):
            new_position = bee[i].Position + np.random.uniform(-ngh, ngh, size=(1, d-1))
            new_position = np.clip(new_position, varMin, varMax)
            new_cost = Cost(new_position)
            if new_cost < bestnewbee.Cost:
                bestnewbee = Bee(new_position, new_cost)
        if bestnewbee.Cost < bee[i].Cost:
            bee[i] = bestnewbee

    # 更新非选择位置的蜜蜂
    for i in range(m, n):
        position = np.random.uniform(varMin, varMax, size=(1, d-1))
        bee[i] = Bee(position, Cost(position))

    bee.sort(key=lambda bee: bee.Cost, reverse=False)
    BestSol = bee[0]
    BestCost[it] = BestSol.Cost
    BestPos[it] = BestSol.Position

    print(f'Iteration {it}: Best Cost = {BestCost[it]}, Best Position = {BestPos[it]}')
    ngh *= 0.95

# 绘制结果
Y = BestCost
X = list(range(len(Y)))
plot.grid(True, which="both")
plot.semilogy(X, Y)
plot.ylim([0, 10])
plot.xlim([0, len(X)])
plot.title('Sphere Function')
plot.xlabel('Iteration')
plot.ylabel('Best Cost')
plot.show()
