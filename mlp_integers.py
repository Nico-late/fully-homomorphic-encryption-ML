"""
In this file we define the basic MLP with weights and biases defined as integers
"""

import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle

class MLP(object):
    def __init__(self, layers=[2, 10, 2], activations=['relu', 'linear'], Q=2**20):
        """"""
        assert (len(layers) == len(activations) + 1)
        self.Q = Q
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.normal(0, int(self.Q*np.sqrt(2 / layers[i])), size=(layers[i + 1], layers[i]
                                                                                                )).astype(int))
            self.biases.append(np.random.normal(0, int(self.Q*np.sqrt(2 / layers[i])), size=(layers[i + 1], 1)
                                                ).astype(int))


    def feedforward(self, x):
        """ return the feedforward value for x """
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            activation_function = self.get_activation_function(self.activations[i])
            z_s.append(self.weights[i].dot(a) // self.Q + self.biases[i])
            a = activation_function(z_s[-1])
            a_s.append(a)
        return z_s, a_s

    def train(self, training_data, labels, val_data=None, val_labels=None, batch_size=32, epochs=100, lr=100, early_stop_steps=10, decay_steps=None, wait_decay_steps=5, decay_threshold=0.01):
        trainin_data_size = len(training_data)
        early_stop = early_stop_steps
        if decay_steps is not None:
            wait_decay = wait_decay_steps
            prev_acc = [0.0]*decay_steps
        for _ in tqdm(range(epochs)):
            mini_batches = [training_data[k:k + batch_size] for k in range(0, trainin_data_size, batch_size)]
            mini_labels = [labels[k:k + batch_size] for k in range(0, trainin_data_size, batch_size)]
            for mini_batch, mini_label in zip(mini_batches, mini_labels):
                self.update_mini_batch(mini_batch, mini_label, lr)
            train_acc = self.get_accuracy(training_data, labels)
            if val_data is not None and val_labels is not None:
                val_acc = self.get_accuracy(val_data, val_labels)
                print('\t train acc:{:.4f} \t val acc:{:.4f}'.format(train_acc, val_acc))
            else:
                print('\t train acc:{:.4f} \t val acc:{:.4f}'.format(train_acc))
            
            if int(train_acc)==1:
                print('\nEarly stop as training accuracy reached 1 !')
                break

            # Learning rate decay
            if decay_steps is not None:
                decay = True
                for i in range(decay_steps):
                    if abs(train_acc-prev_acc[i])>decay_threshold:
                        decay=False
                        break
                if decay:
                    if wait_decay>0:
                        wait_decay-=1
                    else:
                        lr *=2
                        decay_threshold/=2
                        wait_decay=wait_decay_steps
                        print('\nLearning rate increased to {}\n'.format(lr))
                
            # Early stop
            if train_acc==prev_acc[-1]:
                early_stop-=1
                if early_stop<=0:
                    print('\nEarly stop !')
                    break
            else:
                early_stop=early_stop_steps
            
            prev_acc.pop(0)
            prev_acc.append(train_acc)

    def update_mini_batch(self, mini_batch, mini_label, lr):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(mini_batch, mini_label):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w + (dweight // (lr * len(mini_batch))) for w, dweight in
                        zip(self.weights, nabla_w)]
        self.biases = [w + (dbias // (lr * len(mini_batch))) for w, dbias in zip(self.biases, nabla_b)]

    def backprop(self, input, target):
        z_s, a_s = self.feedforward(input)
        deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
        deltas[-1] = ((target - a_s[-1]) * (self.get_derivative_activation_function(self.activations[-1]))(z_s[-1]))
        # Perform BackPropagation
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.weights[i + 1].T.dot(deltas[i + 1]) * (
                self.get_derivative_activation_function(self.activations[i])(z_s[i])) // self.Q
        db = deepcopy(deltas)  # dC/dW
        dw = [d.dot(a_s[i].T) // self.Q for i, d in enumerate(db)]  # dC/dB //self.Q
        # return the derivatives respect to weight matrix and biases
        return db, dw

    @staticmethod
    def get_activation_function(name):
        if name == 'sigmoid':
            return lambda x: np.exp(x) / (1 + np.exp(x))
        elif name == 'linear':
            return lambda x: x
        elif name == 'relu':
            def relu(x):
                y = np.copy(x)
                y[y < 0] = 0
                return y

            return relu
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x

    @staticmethod
    def get_derivative_activation_function(name):
        if name == 'sigmoid':
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))
        elif name == 'linear':
            return lambda x: 1
        elif name == 'relu':
            def relu_diff(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y

            return relu_diff
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1

    def get_accuracy(self, X, Y):
        counter = []
        prediction = self.predict(X)
        for i in range(len(prediction)):
            counter.append(int((Y[i] == prediction[i]).all()))
        return np.sum(counter, axis=0) / Y.shape[0]

    def predict(self, X):
        predict = np.zeros(X.shape).astype(int)
        for i in range(X.shape[0]):
            x = X[i]
            _, a_s = self.feedforward(x)
            ind = np.argmax(a_s[-1])
            predict[i,ind] = [self.Q]
        return predict

    def save_weights(self, full_path):
        with open(full_path, 'wb') as f:
            pickle.dump((self.weights,self.biases), f)

    def load_weights(self, full_path):
        with open(full_path, 'rb') as f:
            self.weights,self.biases = pickle.load(f)