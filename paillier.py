from gmpy2 import random_state, mpz_urandomb, bit_set, next_prime, bit_length, num_digits, mpz_random, gcd, invert, powmod, mpz
import time

## Generate prime numbers using gmpy2
def get_prime(size):
    seed = random_state(time.time_ns())
    p = mpz_urandomb(seed,size)
    p = p.bit_set(size-1) 
    return next_prime(p)

# Fonction qui prend en paramètre la taille de la clé publique et qui retourne la clé publique et privée de Paillier
def get_paillier_keys(size):
    pub_key = priv_key =2
    while gcd(pub_key,priv_key)!=1:
        p = q = 0
        while p==q:
            p = get_prime(size)
            q = get_prime(size)
        pub_key = p*q
        priv_key = (p-1)*(q-1)
    return pub_key, [priv_key,p,q] 

def get_r(pub_key):
    seed = random_state(time.time_ns())
    r = mpz_random(seed,pub_key)
    while gcd(r,pub_key)!=1:
            seed = random_state(time.time_ns())
            r = mpz_random(seed,pub_key)
    return r

# Fonction de chiffrement de Paillier
def paillier_encrypt(message, pub_key, r):
    N_2 = pow(pub_key,2)
    c = (powmod(1+pub_key,message,N_2)*powmod(r,pub_key,N_2))%N_2
    return c