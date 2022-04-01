"""
Main file used to create a object of type Q_class or Q_class_encrypted and test it on data
"""

from Q_class import Q_class
from Q_class_encrypted import Q_class_encrypted
import numpy as np

Q = 64
X_1 = Q * np.array([0,1,0,1])
X_2 = Q * np.array([0,0,1,1])

q_class_object = Q_class(Q,input_type='int')

for i in range(X_1.shape[0]):
    print('Input : {} and {}'.format(X_1[i],X_2[i]))
    print('OR Result : {}'.format(q_class_object.apply_operations(X_1,X_2,'OR')[i][0][0]))
    print('AND Result : {}'.format(q_class_object.apply_operations(X_1,X_2,'AND')[i][0][0]))
    print('XOR Result : {}\n'.format(q_class_object.apply_operations(X_1,X_2,'XOR')[i][0][0]))

q_class_encrypted_object = Q_class_encrypted(Q,input_type='int')

for i in range(X_1.shape[0]):
    print('Input : {} and {}'.format(X_1[i],X_2[i]))
    print('OR Result : {}'.format(q_class_encrypted_object.apply_operations(X_1,X_2,'OR')[i][0][0]))
    print('AND Result : {}'.format(q_class_encrypted_object.apply_operations(X_1,X_2,'AND')[i][0][0]))
    print('XOR Result : {}\n'.format(q_class_encrypted_object.apply_operations(X_1,X_2,'XOR')[i][0][0]))