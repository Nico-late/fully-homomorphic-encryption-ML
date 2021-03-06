"""
This file contains the implementation of the Encrypted Variable used to encrypt
the weights and biases of the MLP_encrypted object defined in mlp_integers_encrypted.py
"""

import numpy as np

class Encrypted_Variable(object):
    def __init__(self, data, pub_key, priv_key):
        self.pub_key = pub_key
        self.priv_key = priv_key
        if isinstance(data, (float,int)):
            self.data = self.pub_key.encrypt(data)
        else:
            self.data = data
    
    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Encrypted_Variable(self.data + other, self.pub_key, self.priv_key)
        else:
            x = self.priv_key.decrypt(other.data)
            return Encrypted_Variable(self.data + x, self.pub_key, self.priv_key)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Encrypted_Variable(self.data - other, self.pub_key, self.priv_key)
        else:
            x = self.priv_key.decrypt(other.data)
            return Encrypted_Variable(self.data - x, self.pub_key, self.priv_key)

    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Encrypted_Variable(self.data * other, self.pub_key, self.priv_key)
        else:
            x = self.priv_key.decrypt(other.data)
            return Encrypted_Variable(self.data * x, self.pub_key, self.priv_key)

    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        r = np.random.randint(64)
        if isinstance(other, (float, int)):
            x=other
        else:
            x = self.priv_key.decrypt(other.data)
        if x == 0:
            raise ZeroDivisionError('Division by zero !')
        y = self.priv_key.decrypt(self.data)
        y = y + r
        y = y // other - r // other
        return Encrypted_Variable(y, self.pub_key, self.priv_key)

    def __floordiv__(self, other):
        return self.__truediv__(other)

    def __str__(self):
        x = self.priv_key.decrypt(self.data)
        return 'encrypted_data={}, decrypted_data= {}'.format(self.data, x)
    
    def __repr__(self):
        return 'Encrypted_Variable' + str(self)
    
    def __neg__(self):
        return Encrypted_Variable(-self.data, self.pub_key, self.priv_key)