# ID: 2018116323 (undergraduate)
# NAME: DaeHeon Yoon
# File name: hw5-1.py
# Platform: Python 3.8.8 on Windows 10
# Required Package(s): sys, os, random, matplotlib, collections, numpy, sklearn


########################### import libraries #####################################
import sys, os
sys.path.append(os.pardir)

import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

import numpy as np
from collections import OrderedDict

from common.optimizer import *
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from common.util import shuffle_dataset

############################## check data ########################################
def load_mnist(normalize=True, one_hot_label=False):
    def _change_one_hot_label(X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
            
        return T

    def train_test_split(data, target, test_size, seed=1004):
        import numpy as np
        
        test_num = int(data.shape[0] * test_size)
        train_num = data.shape[0] - test_num

        np.random.seed(seed)
        shuffled = np.random.permutation(data.shape[0])
        data = data[shuffled,:]
        target = target[shuffled]
        
        x_train = data[:train_num]
        x_test = data[train_num:]
        t_train = target[:train_num]
        t_test = target[train_num:]

        return x_train, x_test, t_train, t_test

    data = load_digits().data
    target = load_digits().target
    

    x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=0.2)
    if normalize:
        x_train = x_train / 16.
        x_test = x_test / 16.

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)  

    return (x_train, t_train), (x_test, t_test)


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
plt.imshow(x_train[random.randint(0, x_train.shape[0])].reshape(8, 8))
plt.show()

########################## optimizer compare naive ###############################
print("\n << optimizer compare naive >>")

def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return x / 10.0, 2.0*y

learning_rates = [[0.95, 0.1, 1.5, 0.3, 0.2], [0.7, 0.05, 0.75, 0.5, 0.1]]

for learning_rate in learning_rates:

    init_pos = (7.0, 2.0)
    params = {}
    params['x'], params['y'] = init_pos[0], init_pos[1]
    grads = {}
    grads['x'], grads['y'] = 0, 0

    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=learning_rate[0])
    optimizers["Momentum"] = Momentum(lr=learning_rate[1])
    optimizers["AdaGrad"] = AdaGrad(lr=learning_rate[2])
    optimizers["Adam"] = Adam(lr=learning_rate[3])
    optimizers["RMSprop"] = RMSprop(lr = learning_rate[4])

    idx = 1

    for key in optimizers:
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]
        
        for i in range(30):
            x_history.append(params['x'])
            y_history.append(params['y'])
            
            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)
        

        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)
        
        X, Y = np.meshgrid(x, y) 
        Z = f(X, Y)
        
        # 외곽선 단순화
        mask = Z > 7
        Z[mask] = 0
        
        # 그래프 그리기
        plt.subplot(3, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color="red")
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        #colorbar()
        #spring()
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")
        
    plt.show()


######################### optimizer compare mnist ################################
print("\n << optimizer compare mnist >>")

# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

learning_rates = [[0.01, 0.01, 0.01, 0.001, 0.01], [0.02, 0.02, 0.02, 0.002, 0.02]]

for learning_rate in learning_rates:

    # 1. 실험용 설정==========
    optimizers = {}
    optimizers["SGD"] = SGD(lr=learning_rate[0])
    optimizers["Momentum"] = Momentum(lr=learning_rate[1])
    optimizers["AdaGrad"] = AdaGrad(lr=learning_rate[2])
    optimizers["Adam"] = Adam(lr=learning_rate[3])
    optimizers["RMSprop"] = RMSprop(lr = learning_rate[4])

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        networks[key] = MultiLayerNet(
            input_size=64, hidden_size_list=[100, 100, 100, 100],
            output_size=10)
        train_loss[key] = []    


    # 2. 훈련 시작==========
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)
        
            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)
        
        if i % 100 == 0:
            print( "===========" + "iteration:" + str(i) + "===========")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))


    # 3. 그래프 그리기==========
    markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D", "RMSprop": "v"}
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


##################### weight init activation histogram ###########################
print("\n << weight init activation histogram >>")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개

w_s = [np.random.randn(node_num, node_num) * 1, np.random.randn(node_num, node_num) * 0.01, np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)]
activation_functions = [sigmoid, ReLU, tanh]

for w in w_s:
    for activation_function in activation_functions:

        input_data = np.random.randn(1000, 100)  # 1000개의 데이터
        activations = {}  # 이곳에 활성화 결과를 저장

        x = input_data

        for i in range(hidden_layer_size):
            if i != 0:
                x = activations[i-1]

            # 초깃값을 다양하게 바꿔가며 실험해보자！
            # w = np.random.randn(node_num, node_num) * 1
            # w = np.random.randn(node_num, node_num) * 0.01
            # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
            # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

            a = np.dot(x, w)

            # 활성화 함수도 바꿔가며 실험해보자！
            z = activation_function(a)
            # z = sigmoid(a)
            # z = ReLU(a)
            # z = tanh(a)

            activations[i] = z

        # 히스토그램 그리기
        for i, a in activations.items():
            plt.subplot(1, len(activations), i+1)
            plt.title(str(i+1) + "-layer")
            if i != 0: plt.yticks([], [])
            # plt.xlim(0.1, 1)
            # plt.ylim(0, 7000)
            plt.hist(a.flatten(), 30, range=(0,1))
        plt.show()


########################## weight init compare ###################################
print("\n << weight init compare >>")

# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

lr_s = [0.01, 0.1]

for lr in lr_s:

    # 1. 실험용 설정==========
    weight_init_types = {'std={}'.format(lr): lr, 'Xavier': 'sigmoid', 'He': 'relu'}
    optimizer = SGD(lr=lr)

    networks = {}
    train_loss = {}
    for key, weight_type in weight_init_types.items():
        networks[key] = MultiLayerNet(input_size=64, hidden_size_list=[100, 100, 100, 100],
                                    output_size=10, weight_init_std=weight_type)
        train_loss[key] = []


    # 2. 훈련 시작==========
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        for key in weight_init_types.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizer.update(networks[key].params, grads)
        
            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)
        
        if i % 100 == 0:
            print("===========" + "iteration:" + str(i) + "===========")
            for key in weight_init_types.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))


    # 3. 그래프 그리기==========
    markers = {'std={}'.format(lr): 'o', 'Xavier': 's', 'He': 'D'}
    x = np.arange(max_iterations)
    for key in weight_init_types.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 2.5)
    plt.legend()
    plt.show()


######################## batch norm gradient check ###############################
print("\n << batch norm gradient check >>")

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = MultiLayerNetExtend(input_size=64, hidden_size_list=[100, 100], output_size=10,
                              use_batchnorm=True)

x_batch = x_train[:1]
t_batch = t_train[:1]

grad_backprop = network.gradient(x_batch, t_batch)
grad_numerical = network.numerical_gradient(x_batch, t_batch)


for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))


########################## batch norm test #######################################
print("\n << batch norm test >>")

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=64, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=64, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# 그래프 그리기==========
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()


######################## overfit weight decay ####################################
print("\n << overfit weight decay >>")

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（가중치 감쇠） 설정 =======================
#weight_decay_lambda = 0 # weight decay를 사용하지 않을 경우
# weight_decay_lambda = 0.1

weight_decay_lambdas = [0.1, 0]

# ====================================================

for weight_decay_lambda in weight_decay_lambdas:

    network = MultiLayerNet(input_size=64, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            weight_decay_lambda=weight_decay_lambda)
    optimizer = SGD(lr=0.01) # 학습률이 0.01인 SGD로 매개변수 갱신

    max_epochs = 201
    train_size = x_train.shape[0]
    batch_size = 100

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break


    # 그래프 그리기==========
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


########################### overfit dropout ######################################
print("\n << overfit dropout >>")

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정 ========================
# use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
use_dropouts = [True, False]
dropout_ratio = 0.2
# ====================================================

for use_dropout in use_dropouts:

    network = MultiLayerNetExtend(input_size=64, hidden_size_list=[100, 100, 100, 100, 100, 100],
                                output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                    epochs=301, mini_batch_size=100,
                    optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
    trainer.train()

    train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

    # 그래프 그리기==========
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


###################### hyperparameter optimization ###############################
print("\n << hyperparameter optimization >>")

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=64, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list

def find_num(num):
    answer = 0
    if 1 <= num < 10:
        return answer
    elif num < 1:
        answer -= 1
        answer *= 10
    elif num >= 10:
        answer += 1
        answer /= 10


# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100

weight_decay_min = -8
weight_decay_max = -4
lr_min = -6
lr_max = -2

for _ in range(3):
    weight_decay_bests = []
    lr_bests = []

    results_val = {}
    results_train = {}
    for __ in range(optimization_trial):
        # 탐색한 하이퍼파라미터의 범위 지정===============
        weight_decay = 10 ** np.random.uniform(weight_decay_min, weight_decay_max)
        lr = 10 ** np.random.uniform(lr_min, lr_max)
        # ================================================

        val_acc_list, train_acc_list = __train(lr, weight_decay)
        print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
        # key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
        key = (lr, weight_decay)

        results_val[key] = val_acc_list
        results_train[key] = train_acc_list

    # 그래프 그리기========================================================
    print("=========== Hyper-Parameter Optimization Result ===========")
    graph_draw_num = 20
    col_num = 5
    row_num = int(np.ceil(graph_draw_num / col_num))
    i = 0

    for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
        print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + "lr:" + str(key[0]) + ", weight decay:" + str(key[1]))
        if (i < 6):
            weight_decay_bests.append(key[0])
            lr_bests.append(key[1])

        plt.subplot(row_num, col_num, i+1)
        plt.title("Best-" + str(i+1))
        plt.ylim(0.0, 1.0)
        if i % 5: plt.yticks([])
        plt.xticks([])
        x = np.arange(len(val_acc_list))
        plt.plot(x, val_acc_list)
        plt.plot(x, results_train[key], "--")
        i += 1

        if i >= graph_draw_num:
            break

    plt.show()

    weight_decay_min = find_num(min(weight_decay_bests))
    weight_decay_max = find_num(max(weight_decay_bests)) + 1
    lr_min = find_num(min(lr_bests))
    lr_max = find_num(max(lr_bests)) + 1

    print("weight decay min:",weight_decay_min, ", weight decay max:", weight_decay_max, ", lr min:", lr_min, ", lr max:", lr_max)
    print()
