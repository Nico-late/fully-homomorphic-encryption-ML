from mlp_integers import MLP
import  os.path
from create_datasets import create_dataset_Q_int,create_dataset_Q_float
import numpy as np

class Q_class(object):
    def __init__(self, Q, input_type='int'):
        self.mlp = {}
        self.operations=['AND','XOR','OR']

        if input_type not in ['int','float']:
            raise ValueError('Accepted input types are float and int !')

        # Create weights directory if doesn't exist
        if not os.path.exists('mlp_weights'):
            os.mkdir('mlp_weights')

        for operation in self.operations:

            # Create operations sub-directories if don't exist
            if not os.path.exists('mlp_weights/{}'.format(operation)):
                os.mkdir('mlp_weights/{}'.format(operation))

            # Load or create MLP model for each operation
            self.mlp[operation] = MLP(layers=[2, 10, 2], activations=['relu', 'linear'], Q=Q)
            path = 'mlp_weights/{}/{}_Q={}_2_10_2.model'.format(operation,input_type,Q)
            if os.path.exists(path):
                self.mlp[operation].load_weights(path)
            else:
                print('##### Model does not exist for operation {} #####'.format(operation))
                print('##### Training of this model #####')
                if input_type=='int':
                    tr_x,tr_y,val_x,val_y = create_dataset_Q_int(Q,1000,operation,validation_split=0.9)
                elif input_type=='float':
                    tr_x,tr_y,val_x,val_y = create_dataset_Q_float(Q,1000,operation,validation_split=0.9)
                self.mlp[operation].train(tr_x, tr_y, val_x, val_y, epochs=100, batch_size=32, lr=100, decay_steps=10)
                self.mlp[operation].save_weights(path)

    # Function that returns the selected operation between the values in X
    def apply_operations(self, X_1, X_2, operation):
        X = np.array([(np.array([x_1, x_2]).reshape(2, 1)) for x_1, x_2 in zip(X_1, X_2)])
        if operation not in self.operations:
            raise ValueError("Avalailable operations are {}".format(self.operations))
        return self.mlp[operation].predict(X)
