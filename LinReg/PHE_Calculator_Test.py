# doesnt work
# can subtract but not usable in rsa for further operations without decryption
# cant divide at all

from lightphe import LightPHE

rsa = None
pal = None

def perform_encryption():
  global rsa, pal, rsa_enc_a, rsa_enc_b, pal_enc_a, pal_enc_b, pal_enc_c

  rsa = LightPHE(algorithm_name="RSA")
  pal = LightPHE(algorithm_name="Paillier")

  a = 20
  b = 5

  rsa_enc_a = rsa.encrypt(a)
  rsa_enc_b = rsa.encrypt(b)

  pal_enc_a = pal.encrypt(a)
  pal_enc_b = pal.encrypt(b)

def multiply():
  global rsa, pal, rsa_enc_a, rsa_enc_b, pal_enc_a, pal_enc_b, pal_enc_c
  print("Multiplication")
  print("RSA: " + str(rsa.decrypt(rsa_enc_a * rsa_enc_b)))

def add():
  global rsa, pal, rsa_enc_a, rsa_enc_b, pal_enc_a, pal_enc_b, pal_enc_c
  print("Addition")
  print("Pallier: " + str(pal.decrypt(pal_enc_a + pal_enc_b)))

def subtract():
  # first finds additive inverse of b by encrypting -1 and multiplying by b using rsa. then uses pallier to add a and additive inverse of b
  global rsa, pal, rsa_enc_a, rsa_enc_b, pal_enc_a, pal_enc_b, pal_enc_c
  print("Subtraction")
  additive_inverse = rsa.encrypt(-1) * rsa_enc_b




perform_encryption()
add()
multiply()
