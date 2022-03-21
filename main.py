from Q_int_class import Q_int_class
import numpy as np

Q = 2**20

#X_1 = np.random.randint(2, size=2)
#X_2 = np.random.randint(2, size=2)
X_1 = np.array([[0],[1],[0],[1]])
X_2 = np.array([[0],[0],[1],[1]])

data = Q * np.array([(np.array([x_1, x_2]).reshape(2, 1)) for x_1, x_2 in zip(X_1, X_2)])
test = Q_int_class(Q)

for i in range(data.shape[0]):
    print("Data n°{}".format(i))
    print('Input {}'.format(list(data[i])))
    print('OR Result : {}'.format(test.apply_operations(data,'OR')[i][0][0]))
    print('AND Result : {}'.format(test.apply_operations(data,'AND')[i][0][0]))
    print('XOR Result : {}\n'.format(test.apply_operations(data,'XOR')[i][0][0]))