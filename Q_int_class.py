from mlp_integers import MLP
import  os.path
from create_datasets import create_dataset_Q_int

class Q_int_class(object):
    def __init__(self, Q):
        self.mlp = {}
        self.operations=['AND','XOR','OR']
        for operation in self.operations:
            self.mlp[operation] = MLP(layers=[2, 10, 2], activations=['relu', 'linear'], Q=Q)
            path = 'mlp_weights/{}/mlp_weights_Q={}_2_10_2.model'.format(operation,Q)
            if os.path.isfile(path):
                self.mlp[operation].load_weights(path)
            else:
                print('##### Model does not exist for operation {} #####'.format(operation))
                print('##### Training of this model #####')
                tr_x,tr_y,val_x,val_y = create_dataset_Q_int(Q,1000,operation,validation_split=0.9)
                self.mlp[operation].train(tr_x, tr_y, val_x, val_y, epochs=100, batch_size=32, lr=100, decay_steps=10)
                self.mlp[operation].save_weights(path)

    def apply_operations(self, X, operation):
        return self.mlp[operation].predict(X)
