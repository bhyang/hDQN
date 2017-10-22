from __future__ import division
import random
import numpy as np
import matplotlib.pyplot as plt

from delayed_reward import game

env = game()

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Initialize experience replays and Q-networks
D1, D2 = [], []

Q1 = Sequential([Dense(2, input_dim=12,bias=False, weights=[0.01*np.random.uniform(size=[12,2])])])
Q1.compile(optimizer=SGD(lr=0.1), loss='mse')

Q2 = Sequential([Dense(6, input_dim=6,bias=False, weights=[0.01*np.random.uniform(size=[6,6])])])
Q2.compile(optimizer=SGD(lr=0.1), loss='mse')

# Learning parameters
y = .99
e = 1
num_episodes = 1000
batch_size = 32

action_space = [0,1]
goal_space = [0,1,2,3,4,5]

jList, rList = [], []

def epsGreedy(x, B, epsilon, Q):
    if np.random.rand(1) < epsilon:
        return random.choice(B)
    else:
        return np.argmax(Q.predict(x))

def onehot(s):
    return np.identity(6)[s:s+1]

def updateParams(Q, D):
    if len(D) >= batch_size:
        minibatch = random.sample(D, batch_size)
        # s0, a, r, s1 = random.choice(D)
        # Qvalues = Q.predict(s1)
        # maxQ = np.max(Qvalues)
        # targetQ = allQ
        # targetQ[0, a] = r + y*maxQ
        # Q.fit(s1, targetQ, verbose=False)
        for s0, action, reward, s1, done in minibatch:
            target = reward
            if not done:
                target = reward + (y*np.amax(Q.predict(s1)[0]))

            target_f = Q.predict(s0)
            target_f[0][action] = target

            Q.fit(s0, target_f, epochs=1, verbose=0)

for i in range(num_episodes):
    s = env.reset()[0]-1
    done = False
    counter = 0    

    goal = epsGreedy(onehot(s), goal_space, e, Q2)

    while not done:
        goalDone = False
        extrinsicTotal = 0
        s0 = s

        while not (done or goalDone):
            counter += 1
            # print(s, goal)
            # print(onehot(s), onehot(goal))
            sg = np.concatenate([onehot(s), onehot(goal)], axis=1)
            action = epsGreedy(sg, action_space, e, Q1)
            s1, r, done = env.moveAndReturn(action)
            s1 -= 1
            extrinsicTotal += r
            # print(s1, goal)
            # print(onehot(s1), onehot(goal))
            sg1 = np.concatenate([onehot(s1), onehot(goal)], axis=1)

            s = s1
            intrinsic_r = 0
            if s1 == goal:
                # print(goal)
                intrinsic_r += 1
                goalDone = True
            
            D1.append((sg, action, intrinsic_r, sg1, goalDone))
            updateParams(Q1, D1)
            updateParams(Q2, D2)
        
        D2.append((onehot(s0), goal, extrinsicTotal, onehot(s1), done))
        if not done:
            goal = epsGreedy(onehot(s), goal_space, e, Q2)

    jList.append(counter)
    rList.append(extrinsicTotal)
    print('Reward: '+ str(extrinsicTotal))
    e *= .993
    # print(e)

print('Average reward: ' + str(sum(rList) / len(rList)))
plt.plot(rList)
plt.show()
plt.plot(jList)
plt.show()