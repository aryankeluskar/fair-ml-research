import random
from sympy import mod_inverse
from lightphe import LightPHE

class ElGamalHomomorphic:
    def __init__(self, p, g):
        self.p = p  # A prime number
        self.g = g  # Generator
        self.private_key = random.randint(2, p - 2)  # Private key
        self.public_key = pow(g, self.private_key, p)  # Public key

    def encrypt(self, plaintext):
        y = random.randint(2, self.p - 2)  # Random value for each encryption
        c1 = pow(self.g, y, self.p)
        s = pow(self.public_key, y, self.p)
        c2 = (plaintext * s) % self.p
        return c1, c2

    def decrypt(self, ciphertext):
        c1, c2 = ciphertext
        s = pow(c1, self.private_key, self.p)
        plaintext = (c2 * mod_inverse(s, self.p)) % self.p
        return plaintext

    def homomorphic_addition(self, eg_cipher1, eg_cipher2):
        c1_1, c2_1 = eg_cipher1
        c1_2, c2_2 = eg_cipher2

        # Perform addition homomorphically
        c1_add = (c1_1 * c1_2) % self.p
        c2_add = (c2_1 * c2_2) % self.p

        return c1_add, c2_add

    def homomorphic_subtraction(self, eg_cipher1, eg_cipher2):
        c1_1, c2_1 = eg_cipher1
        c1_2, c2_2 = eg_cipher2

        # Perform subtraction homomorphically
        c1_sub = (c1_1 * mod_inverse(c1_2, self.p)) % self.p
        c2_sub = (c2_1 * mod_inverse(c2_2, self.p)) % self.p

        return c1_sub, c2_sub

    def homomorphic_multiplication(self, eg_cipher1, scalar):
        c1, c2 = eg_cipher1

        # Perform multiplication homomorphically
        c1_mul = pow(c1, scalar, self.p)
        c2_mul = pow(c2, scalar, self.p)

        return c1_mul, c2_mul

    def homomorphic_division(self, eg_cipher1, eg_cipher2):
        c1_1, c2_1 = eg_cipher1
        c1_2, c2_2 = eg_cipher2

        # Perform division homomorphically
        c1_div = (c1_1 * mod_inverse(c1_2, self.p)) % self.p
        c2_div = (c2_1 * mod_inverse(c2_2, self.p)) % self.p

        return c1_div, c2_div

    def homomorphic_exponentiation(self, ciphertext, exponent):
        c1, c2 = ciphertext

        # Perform exponentiation homomorphically
        c1_exp = pow(c1, exponent, self.p)
        c2_exp = pow(c2, exponent, self.p)

        return c1_exp, c2_exp

# Example usage:
elgamal = ElGamalHomomorphic(p=61, g=random.randint(2, 59))

plaintext1 = 42
plaintext2 = 7

# Encryption
eg_cipher1 = elgamal.encrypt(plaintext1)
eg_cipher2 = elgamal.encrypt(plaintext2)
pal_cipher1 

# Homomorphic Operations
result_add = elgamal.homomorphic_addition(eg_cipher1, eg_cipher2)
result_sub = elgamal.homomorphic_subtraction(eg_cipher1, eg_cipher2)
result_mul = elgamal.homomorphic_multiplication(eg_cipher1, scalar=3)
result_div = elgamal.homomorphic_division(eg_cipher1, eg_cipher2)
result_exp = elgamal.homomorphic_exponentiation(eg_cipher1, exponent=2)

# Decryption
result_add_decrypted = elgamal.decrypt(result_add)
result_sub_decrypted = elgamal.decrypt(result_sub)
result_mul_decrypted = elgamal.decrypt(result_mul)
result_div_decrypted = elgamal.decrypt(result_div)
result_exp_decrypted = elgamal.decrypt(result_exp)

print(f"Original 1: {plaintext1}")
print(f"Original 2: {plaintext2}")
print(f"Addition Result: {result_add_decrypted}")
print(f"Subtraction Result: {result_sub_decrypted}")
print(f"Multiplication Result: {result_mul_decrypted}")
print(f"Division Result: {result_div_decrypted}")
print(f"Exponentiation Result: {result_exp_decrypted}")
