from Q_class import Q_class
from Q_class_encrypted import Q_class_encrypted
import numpy as np

Q = 2**20

#X_1 = Q * np.random.randint(2, size=2)
#X_2 = Q * np.random.randint(2, size=2)
X_1 = Q * np.array([0,1,0,1])
X_2 = Q * np.array([0,0,1,1])

test = Q_class_encrypted(Q,input_type='int')

for i in range(X_1.shape[0]):
    print('Input : {} and {}'.format(X_1[i],X_2[i]))
    print('OR Result : {}'.format(test.apply_operations(X_1,X_2,'OR')[i][0][0]))
    print('AND Result : {}'.format(test.apply_operations(X_1,X_2,'AND')[i][0][0]))
    print('XOR Result : {}\n'.format(test.apply_operations(X_1,X_2,'XOR')[i][0][0]))