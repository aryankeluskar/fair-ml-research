import tenseal as ts

context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)

a = 10
b = 3

a_enc = ts.bfv_vector(context, [a])
b_enc = ts.bfv_vector(context, [b])

def add():
    global a_enc, b_enc
    return a_enc + b_enc

def subtract():
    global a_enc, b_enc
    return a_enc - b_enc

def multiply():
    global a_enc, b_enc
    return a_enc * b_enc

def divide():
    global a_enc, b_enc
    # use repititive subtraction to divide

    # first find additive inverse of b
    additive_inverse = -1 * b_enc
    for i in range():
        additive_inverse += b_enc


print(add()._decrypt())
print(subtract()._decrypt())
print(multiply()._decrypt())
print(divide()._decrypt())