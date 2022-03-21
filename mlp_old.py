# -*- coding: utf-8 -*-
"""
  @Author: zzn
  @Date: 2019-01-02 14:36:33
  @Last Modified by:   nico-late
  @Last Modified time: 2022-01-03 15:48:15
"""

import numpy as np
import pickle
from tqdm import tqdm

def relu(x):
    return np.maximum(0,x)
    
def linear(x):
    return x
    
def relu_derivative(x):
    return (x > 0) * 1

def linear_derivative(x):
    return 1
    
def attribute_activation_function(activation_function):
    if activation_function == 'relu':
        return relu
    elif activation_function == 'linear':
        return linear
    else:
        raise ValueError('Activation function is not defined.')

def get_derivative_function(activation_function):
    if activation_function == 'relu':
        return relu_derivative
    elif activation_function == 'linear':
        return linear_derivative
    else:
        raise ValueError('Activation function is not defined.')
    

class MLP_float(object):
    def __init__(self, in_size=10, hidden_sizes=(100,),hidden_activation='relu', out_activation='linear'):
        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes)+2
        self.parameters = self.initializing_paras()
        self.hidden_activation = attribute_activation_function(hidden_activation)
        self.hidden_derivative = get_derivative_function(hidden_activation)
        self.out_activation = attribute_activation_function(out_activation)
        self.out_derivative = get_derivative_function(out_activation)

    def initializing_paras(self):
        paras = {}
        in_dim = self.in_size
        for i, h_dim in enumerate(self.hidden_sizes):
            out_dim = h_dim
            w_tmp = np.random.normal(size=(in_dim, out_dim))
            b_tmp = np.random.normal(size=(out_dim))
            paras['W_{}'.format(i)] = w_tmp
            paras['b_{}'.format(i)] = b_tmp
            in_dim = out_dim
        out_dim = 1
        w_tmp = np.random.normal(size=(in_dim, out_dim))
        b_tmp = np.random.normal(size=(out_dim))
        paras['W_{}'.format(i+1)] = w_tmp
        paras['b_{}'.format(i+1)] = b_tmp
        return paras

    def save_weights(self, full_path):
        with open(full_path, 'wb') as f:
            pickle.dump(self.parameters, f)

    def load_weights(self, full_path):
        with open(full_path, 'rb') as f:
            self.parameters = pickle.load(f)

    def predict(self, x, batch_size=32):
        n, d = x.shape
        output = np.zeros((n,1))
        for i in range(0, n, batch_size):
            batch_x = x[i:i+batch_size]
            batch_y, _ = self.forward_batch(batch_x)
            output[i:i+batch_size] = batch_y
        output = (output>0.5) * 1
        return np.squeeze(output)

    def forward_batch(self, batch_x):
        batch_size = batch_x.shape[0]
        out_list = []
        out_list.append(batch_x)
        for i in range(self.n_layers-1):
            w_tmp = self.parameters['W_{}'.format(i)]
            b_tmp = self.parameters['b_{}'.format(i)]
            batch_x = np.tile(b_tmp,(batch_size,1)) + np.dot(batch_x, w_tmp)
            if i != self.n_layers-2:
                batch_x = self.hidden_activation(batch_x)
                out_list.append(batch_x)
            else:
                batch_x = self.out_activation(batch_x)
                out_list.append(batch_x)
        batch_y = batch_x
        return batch_y, out_list

    def backward_batch(self, batch_y_true, output_list):
        param_grads = {}
        for i in range(self.n_layers-1, 0, -1):
            output = output_list[i]
            h = output_list[i-1]
            if i == self.n_layers-1:
                dL_ds = (output-batch_y_true)*self.out_derivative(output-batch_y_true)
                tmp = dL_ds
            else:
                dL_ds = output*self.hidden_derivative(output)*np.dot(tmp,self.parameters['W_{}'.format(i)].T)
                tmp = dL_ds
            ds_dw = h
            dL_dw = np.dot(ds_dw.T, dL_ds)
            dL_db = dL_ds.sum(axis=0)
            param_grads['W_{}'.format(i-1)] = dL_dw
            param_grads['b_{}'.format(i-1)] = dL_db
        return param_grads

    def update_params_sgd(self, param_grads, lr, v, momentum=0):
        for _, key in enumerate(param_grads):
            v[key] = momentum*v[key]-lr*param_grads[key]
            self.parameters[key] += v[key]
        return v
        
    def update_params_adam(self, param_grads, lr, v, m, epoch, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for _, key in enumerate(param_grads):
            ## momentum beta 1
            m[key] = beta1*m[key] + (1-beta1)*param_grads[key]

            ## rms beta 2
            v[key] = beta2*v[key] + (1-beta2)*(param_grads[key]**2)

            ## bias correction
            m_corr = m[key]/(1-beta1**(epoch+1))
            v_corr = v[key]/(1-beta2**(epoch+1))

            ## update weights and biases
            self.parameters[key] -= (lr*m_corr/(np.sqrt(v_corr)+epsilon))
        return v,m

    def fit(self, x, y, val_x=None, val_y=None, batch_size=32, epochs=100, lr=1e-2, decay_steps=5, wait_decay_steps=5, early_stop_steps=10,optimizer='adam'):
        n = x.shape[0]
        indexs = np.arange(x.shape[0])
        prev_train_acc = 0
        decay,early_stop,wait_decay,decay_threshold = decay_steps,early_stop_steps,wait_decay_steps,0.001
        for e in tqdm(range(1, epochs+1)):
            np.random.shuffle(indexs)
            x = x[indexs]
            y = y[indexs]
            for i in range(0, n, batch_size):
                batch_x = x[i:i+batch_size]
                batch_y_true = np.expand_dims(y[i:i+batch_size],axis=1)
                _, out_list = self.forward_batch(batch_x)
                param_grads = self.backward_batch(batch_y_true, out_list)
                if optimizer=='sgd':
                    if e == 1 and i == 0:
                        v = {key: 0
                             for i, key in enumerate(param_grads)}
                    else:
                        v = self.update_params_sgd(
                            param_grads, lr=lr, v=v)
                elif optimizer=='adam':
                    if e == 1 and i == 0:
                        v = {key: 0
                             for _, key in enumerate(param_grads)}
                        m = {key: 0
                             for _, key in enumerate(param_grads)}
                    else:
                        v,m = self.update_params_adam(
                            param_grads, lr=lr, v=v, m=m, epoch=e)
                else:
                    raise ValueError('Optimizer is not defined.')
                    
            pred_y = self.predict(x)
            val_pred_y = self.predict(val_x)
            train_acc = np.mean(pred_y == y)
            val_acc = np.mean(val_pred_y == val_y)

            print('\t train acc:{:.4f} \t val acc:{:.4f}'.format(train_acc, val_acc))
            
            # Accuracy reached one so early stop
            if int(train_acc)==1:
                print('\nEarly stop as training accuracy reached 1 !')
                break

            # Learning rate decay
            if abs(train_acc-prev_train_acc)<=decay_threshold:
                if wait_decay>0:
                    wait_decay-=1
                else:
                    decay-=1
                    if decay<=0:
                        lr *=1e-1
                        decay_threshold/=2
                        decay=decay_steps
                        wait_decay=wait_decay_steps
                        print('\nLearning rate decreased to {}\n'.format(lr))
            else:
                decay=decay_steps
                
            # Early stop
            if train_acc==prev_train_acc:
                early_stop-=1
                if early_stop<=0:
                    print('\nEarly stop !')
                    break
            else:
                early_stop=early_stop_steps
                
            prev_train_acc = train_acc