"""
This file defines all the function used to automatically generate the datasets used in the MLPs
"""

import numpy as np

def apply_operation(x1,x2,operation):
    if operation == 'AND':
        operation_value = int(np.logical_and(x1, x2))
    elif operation == 'XOR':
        operation_value = int(np.logical_xor(x1, x2))
    elif operation == 'OR':
        operation_value = int(np.logical_or(x1, x2))
    else:
        operations=['AND','XOR','OR']
        raise ValueError('Operation not implemented ! Operations available: {}'.format(operations))
    return [operation_value, 1 - operation_value]

def create_dataset_Q_int(Q, dataset_size=10000,operation='XOR',validation_split=0):
    
    if type(validation_split) not in [float,int] or validation_split<0 or validation_split>1:
        raise ValueError('Validation_split has to be a float between 0 and 1.')
    
    X_1 = np.random.randint(2, size=dataset_size)
    X_2 = np.random.randint(2, size=dataset_size)


    data = Q * np.array([(np.array([x_1, x_2]).reshape(2, 1)) for x_1, x_2 in zip(X_1, X_2)])
    labels = Q * np.array([np.array(apply_operation(x_1,x_2,operation)).reshape(2, 1) for x_1, x_2 in zip(X_1, X_2)])
    if validation_split!=0:
        split = int(dataset_size*validation_split)
        return data[:split],labels[:split],data[split:],labels[split:]
    return data,labels

def create_dataset_Q_float(Q, dataset_size=10000,operation='XOR',validation_split=0):

    if type(validation_split) not in [float,int] or validation_split<0 or validation_split>1:
        raise ValueError('Validation_split has to be a float between 0 and 1.')
    
    X_1 = np.random.random(size=dataset_size)
    X_2 = np.random.random(size=dataset_size)


    data = Q * np.array([(np.array([x_1, x_2]).reshape(2, 1)) for x_1, x_2 in zip(X_1, X_2)])
    labels = Q * np.array([np.array(apply_operation(round(x_1),round(x_2),operation)).reshape(2, 1) for x_1, x_2 in zip(X_1, X_2)])
    if validation_split!=0:
        split = int(dataset_size*validation_split)
        return data[:split],labels[:split],data[split:],labels[split:]
    return data,labels