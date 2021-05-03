# ID: 2018116323 (undergraduate)
# NAME: DaeHeon Yoon
# File name: hw6.py
# Platform: Python 3.8.8 on Windows 10
# Required Package(s): numpy, sklearn


import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from collections import OrderedDict

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


from common.layers import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

# -----------------------------------------------------------------------------
# modified from train_neuralnet.py

def train_neuralnet_mnist(x_train, t_train, x_test, t_test, 
                          input_size=64, hidden_size=10, output_size=10, 
                          iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                          verbose=True):
    
    network = TwoLayerNet(input_size, hidden_size, output_size)

    # Train Parameters
    train_size = x_train.shape[0]
    iter_per_epoch = int(max(train_size / batch_size, 1))

    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for step in range(1, iters_num+1):
        # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
        grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(압도적으로 빠르다)

        # Update
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # loss
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if verbose and step % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('Step: {:4d}\tTrain acc: {:.5f}\tTest acc: {:.5f}'.format(step, 
                                                                            train_acc,
                                                                            test_acc))
    tracc, teacc = network.accuracy(x_train, t_train), network.accuracy(x_test, t_test)
    if verbose:
        print('Optimization finished!')
        print('Training accuracy: %.2f' % tracc)
        print('Test accuracy: %.2f' % teacc)
    return tracc, teacc

#--------------------------------------------------------------------------------
print("######################## Incorrect holdout split ######################### ")
def load_mnist1(normalize=True, one_hot_label=False, shuffled=True):
    def _change_one_hot_label(X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
            
        return T

    def train_test_split(data, target, test_size, shuffled=True, seed=1004):
        import numpy as np
        
        test_num = int(data.shape[0] * test_size)
        train_num = data.shape[0] - test_num

        if shuffled:
            np.random.seed(seed)
            shuffled = np.random.permutation(data.shape[0])
            data = data[shuffled,:]
            target = target[shuffled]
        else:
            idx = np.argsort(target)
            data = data[idx]
            target = target[idx]

        x_train = data[:train_num]
        x_test = data[train_num:]
        t_train = target[:train_num]
        t_test = target[train_num:]

        return x_train, x_test, t_train, t_test

    data = load_digits().data
    target = load_digits().target
    

    x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=0.4, shuffled = shuffled)
    if normalize:
        x_train = x_train / 16.
        x_test = x_test / 16.

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)  

    return (x_train, t_train), (x_test, t_test)

(x_train, t_train), (x_test, t_test) = load_mnist1(shuffled=False)

train_neuralnet_mnist(x_train, t_train, x_test, t_test, 
                     input_size=64, hidden_size=10, output_size=10, 
                     iters_num = 3000, batch_size = 10, learning_rate = 0.1)


#----------------------------------------------------------------------------------
print("#################### Incorrect holdout split ##########################")

def load_mnist2(normalize=True, one_hot_label=False, shuffled=True):
    def _change_one_hot_label(X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
            
        return T

    def train_test_split(data, target, test_size, shuffled=True, seed=1004):
        import numpy as np
        
        test_num = int(data.shape[0] * test_size)
        train_num = data.shape[0] - test_num

        if shuffled:
            np.random.seed(seed)
            shuffled = np.random.permutation(data.shape[0])
            data = data[shuffled,:]
            target = target[shuffled]
        else:
            idx = np.argsort(target)
            data = data[idx]
            target = target[idx]

        idx = np.where(target == 0)
        data_split = data[idx]
        target_split = target[idx]

        test_num = int(data_split.shape[0] * test_size)
        train_num = data_split.shape[0] - test_num

        x_train = data_split[:train_num]
        x_test = data_split[train_num:]
        t_train = target_split[:train_num]
        t_test = target_split[train_num:]

        for i in range(1,10):
            idx = np.where(target == i)
            data_split = data[idx]
            target_split = target[idx]

            test_num = int(data_split.shape[0] * 0.4)
            train_num = data_split.shape[0] - test_num

            x_train = np.append(x_train, data_split[:train_num], axis=0)
            x_test = np.append(x_test, data_split[train_num:], axis=0)
            t_train = np.append(t_train, target_split[:train_num], axis=0)
            t_test = np.append(t_test, target_split[train_num:], axis=0)
        


        # x_train = data[:train_num]
        # x_test = data[train_num:]
        # t_train = target[:train_num]
        # t_test = target[train_num:]

        return x_train, x_test, t_train, t_test

    data = load_digits().data
    target = load_digits().target
    

    x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=0.4, shuffled = shuffled)
    if normalize:
        x_train = x_train / 16.
        x_test = x_test / 16.

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)  

    return (x_train, t_train), (x_test, t_test)

(x_train, t_train), (x_test, t_test) = load_mnist2(shuffled=False)


train_neuralnet_mnist(x_train, t_train, x_test, t_test, 
                     input_size=64, hidden_size=10, output_size=10, 
                     iters_num = 3000, batch_size = 10, learning_rate = 0.1)


#-------------------------------------------------------------------------------------
print("#################### holdout split by random sampling #####################")


def load_mnist3(normalize=True, one_hot_label=False, shuffled=True):
    def _change_one_hot_label(X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
            
        return T

    def train_test_split(data, target, test_size, shuffled=True, seed=1004):
        import numpy as np
        
        test_num = int(data.shape[0] * test_size)
        train_num = data.shape[0] - test_num

        if shuffled:
            np.random.seed(seed)
            shuffled = np.random.permutation(data.shape[0])
            data = data[shuffled,:]
            target = target[shuffled]
        else:
            idx = np.argsort(target)
            data = data[idx]
            target = target[idx]

        idx = np.where(target == 0)
        data_split = data[idx]
        target_split = target[idx]
        
        np.random.seed(seed)
        shuffled = np.random.permutation(data_split.shape[0])
        data_split = data_split[shuffled,:]
        target_split = target_split[shuffled]

        test_num = int(data_split.shape[0] * test_size)
        train_num = data_split.shape[0] - test_num

        x_train = data_split[:train_num]
        x_test = data_split[train_num:]
        t_train = target_split[:train_num]
        t_test = target_split[train_num:]

        for i in range(1,10):
            idx = np.where(target == i)
            data_split = data[idx]
            target_split = target[idx]

            np.random.seed(seed)
            shuffled = np.random.permutation(data_split.shape[0])
            data_split = data_split[shuffled,:]
            target_split = target_split[shuffled]

            test_num = int(data_split.shape[0] * 0.4)
            train_num = data_split.shape[0] - test_num

            x_train = np.append(x_train, data_split[:train_num], axis=0)
            x_test = np.append(x_test, data_split[train_num:], axis=0)
            t_train = np.append(t_train, target_split[:train_num], axis=0)
            t_test = np.append(t_test, target_split[train_num:], axis=0)
    


        # x_train = data[:train_num]
        # x_test = data[train_num:]
        # t_train = target[:train_num]
        # t_test = target[train_num:]

        return x_train, x_test, t_train, t_test

    data = load_digits().data
    target = load_digits().target
    

    x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=0.4, shuffled = shuffled)
    if normalize:
        x_train = x_train / 16.
        x_test = x_test / 16.

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)  

    return (x_train, t_train), (x_test, t_test)

(x_train, t_train), (x_test, t_test) = load_mnist3(shuffled=False)


train_neuralnet_mnist(x_train, t_train, x_test, t_test, 
                     input_size=64, hidden_size=10, output_size=10, 
                     iters_num = 3000, batch_size = 10, learning_rate = 0.1)




#-----------------------------------------------------------------------------------------
print("################# sklearn의 train_test_split 사용 ########################")

data = load_digits().data
target = load_digits().target

x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=0.4, shuffle=True)

x_train = x_train / 16.
x_test = x_test / 16.

train_neuralnet_mnist(x_train, t_train, x_test, t_test, 
                     input_size=64, hidden_size=10, output_size=10, 
                     iters_num = 3000, batch_size = 10, learning_rate = 0.1)

#----------------------------------------------------------------------------------------
print("################### Reproducible random sampling #####################")

# HO4: Reproducible Random Sampling
# Random sampling by sklearn.model_selection.train_test_split
# source: https://scikit-learn.org/stable/modules/cross_validation.html


data = load_digits().data
target = load_digits().target

x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=0.4, shuffle=True, random_state=len(target))

np.random.seed(len(target))

x_train = x_train / 16.
x_test = x_test / 16.

# fix the SEED of random permutation to be the number of samples, 
# to reproduce the same random sequence at every execution
np.random.seed(len(target))

train_neuralnet_mnist(x_train, t_train, x_test, t_test,
                     input_size=64, hidden_size=10, output_size=10, 
                     iters_num = 1000, batch_size = 10, learning_rate = 0.1)

#---------------------------------------------------------------------------------------
print("####################### Stratified random sampling ######################")

# HO5: Stratified Random Sampling}
X = load_digits().data
y = load_digits().target
X = X / 16.

# per-class random sampling by passing y to variable stratify, 
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y)

# check number of samples of the individual classes
print('test: %d %d %d %d %d %d %d %d %d %d,  '%(sum(yte==0),sum(yte==1),sum(yte==2),sum(yte==3),sum(yte==4),sum(yte==5),sum(yte==6),sum(yte==7),sum(yte==8),sum(yte==9)),end='')
print('training: %d %d %d %d %d %d %d %d %d %d,  '%(sum(ytr==0),sum(ytr==1),sum(ytr==2),sum(ytr==3),sum(ytr==4),sum(ytr==5),sum(ytr==6),sum(ytr==7),sum(ytr==8),sum(ytr==9)))
# due to the random initialization of the weights, the performance varies
# so we have to set the random seed for TwoLayerNet's initialization values
np.random.seed(len(y))

train_neuralnet_mnist(Xtr,ytr,Xte,yte,
                     input_size=64, hidden_size=10, output_size=10, 
                     iters_num = 1000, batch_size = 10, learning_rate = 0.1)

#---------------------------------------------------------------------------------------
print("###################### Repeated random subsampling #######################")
# Repeated Random Subsampling
# Repeating stratified random sampling K times


X = load_digits().data
y = load_digits().target
X = X / 16.

# due to the random initialization of the weights, the performance varies
# so we have to set the random seed for TwoLayerNet's initialization values
np.random.seed(len(y))

K = 20
Acc = np.zeros([K,2], dtype=float)
for k in range(K):
    # stratified random sampling
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=None, stratify=y)
    Acc[k,0], Acc[k,1] = train_neuralnet_mnist(Xtr,ytr,Xte,yte,
                                  input_size=64, hidden_size=10, output_size=10, 
                                  iters_num = 1000, batch_size = 10, learning_rate = 0.1, 
                                  verbose = False)
    print('Trial %d: accuracy %.3f %.3f' % (k, Acc[k,0], Acc[k,1]))
