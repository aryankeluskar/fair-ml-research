import torch
import tenseal as ts
import pandas as pd
import random
from time import time

# those are optional and are not necessary for training
import numpy as np
import matplotlib.pyplot as plt

def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    # delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]


def heart_disease_data():
    data = pd.read_csv("./framingham.csv")
    # drop rows with missing values
    data = data.dropna()
    # drop some features
    data = data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])
    # balance data
    grouped = data.groupby('TenYearCHD')
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    # extract labels
    y = torch.tensor(data["TenYearCHD"].values).float().unsqueeze(1)
    data = data.drop("TenYearCHD", axis='columns')
    # standardize data
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    return split_train_test(x, y)


def random_data(m=1024, n=2):
    # data separable by the line `y = x`
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)
    y_train = (x_train[:, 0] >= x_train[:, 1]).float().unsqueeze(0).t()
    y_test = (x_test[:, 0] >= x_test[:, 1]).float().unsqueeze(0).t()
    return x_train, y_train, x_test, y_test


# You can use whatever data you want without modification to the tutorial
# x_train, y_train, x_test, y_test = random_data()
x_train, y_train, x_test, y_test = heart_disease_data()

print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print(f"y_train has shape: {y_train.shape}")
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################")

 
# ## Training a Logistic Regression Model
# 
# We will start by training a logistic regression model (without any encryption), which can be viewed as a single layer neural network with a single node. We will be using this model as a means of comparison against encrypted training and evaluation.

 
class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out

 
n_features = x_train.shape[1]
model = LR(n_features)
# use gradient descent with a learning_rate=1
optim = torch.optim.SGD(model.parameters(), lr=1)
# use Binary Cross Entropy Loss
criterion = torch.nn.BCELoss()

 
# define the number of epochs for both plain and encrypted training
EPOCHS = 5

def train(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print(f"Loss at epoch {e}: {loss.data}")
    return model

model = train(model, optim, criterion, x_train, y_train)

 
def accuracy(model, x, y):
    out = model(x)
    correct = torch.abs(y - out) < 0.5
    return correct.float().mean()

plain_accuracy = accuracy(model, x_test, y_test)
print(f"Accuracy on plain test_set: {plain_accuracy}")

 
# It is worth to remember that a high accuracy isn't our goal. We just want to see that training on encrypted data doesn't affect the final result, so we will be comparing accuracies over encrypted data against the `plain_accuracy` we got here.

 
# ## Encrypted Evaluation
# 
# In this part, we will just focus on evaluating the logistic regression model with plain parameters (optionally encrypted parameters) on the encrypted test set. We first create a PyTorch-like LR model that can evaluate encrypted data:

 
class EncryptedLR:
    
    def __init__(self, torch_lr):
        # TenSEAL processes lists and not torch tensors,
        # so we take out the parameters from the PyTorch model
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        
    def forward(self, enc_x):
        # We don't need to perform sigmoid as this model
        # will only be used for evaluation, and the label
        # can be deduced without applying sigmoid
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    ################################################
    ## You can use the functions below to perform ##
    ## the evaluation with an encrypted model     ##
    ################################################
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self, context):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()
        

eelr = EncryptedLR(model)

 
# We now create a TenSEALContext for specifying the scheme and the parameters we are going to use. Here we choose small and secure parameters that allow us to make a single multiplication. That's enough for evaluating a logistic regression model, however, we will see that we need larger parameters when doing training on encrypted data.

 
# parameters
poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
# create TenSEALContext
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
# scale of ciphertext to use
ctx_eval.global_scale = 2 ** 20
# this key is needed for doing dot-product operations
ctx_eval.generate_galois_keys()

 
# We will encrypt the whole test set before the evaluation:

 
t_start = time()
enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
t_end = time()
print(f"Encryption of the test-set took {int(t_end - t_start)} seconds")

 
# (optional) encrypt the model's parameters
# eelr.encrypt(ctx_eval)

 
# As you may have already noticed when we built the EncryptedLR class, we don't compute the sigmoid function on the encrypted output of the linear layer, simply because it's not needed, and computing sigmoid over encrypted data will increase the computation time and require larger encryption parameters. However, we will use sigmoid for the encrypted training part. We now proceed with the evaluation of the encrypted test set and compare the accuracy to the one on the plain test set.

 
def encrypted_evaluation(model, enc_x_test, y_test):
    t_start = time()
    
    correct = 0
    for enc_x, y in zip(enc_x_test, y_test):
        # encrypted evaluation
        enc_out = model(enc_x)
        # plain comparison
        out = enc_out.decrypt()
        out = torch.tensor(out)
        out = torch.sigmoid(out)
        if torch.abs(out - y) < 0.5:
            correct += 1
    
    t_end = time()
    print(f"Evaluated test_set of {len(x_test)} entries in {int(t_end - t_start)} seconds")
    print(f"Accuracy: {correct}/{len(x_test)} = {correct / len(x_test)}")
    return correct / len(x_test)
    

encrypted_accuracy = encrypted_evaluation(eelr, enc_x_test, y_test)
diff_accuracy = plain_accuracy - encrypted_accuracy
print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")
if diff_accuracy < 0:
    print("Oh! We got a better accuracy on the encrypted test-set! The noise was on our side...")

 
# We saw that evaluating on the encrypted test set doesn't affect the accuracy that much. I've even seen examples where the encrypted evaluation performs better.

 
# ## Training an Encrypted Logistic Regression Model on Encrypted Data
# 
# In this part we will redefine a PyTorch-like model that can both forward encrypted data, as well as backpropagate to update the weights and thus train the encrypted logistic regression model on encrypted data. Below are more details about the training.
# 
# #### Loss Function
# 
# We are using the binary cross entropy loss function with regularization (more about the why of regularization will follow) where $y^{(i)}$ is the i'th expected label, $\hat{y}^{(i)}$ is the i'th output of the logistic regression model and $\theta$ is our n-sized weight vector.
# 
# $$Loss(\theta) = - \frac{1}{m} \sum_{i=1}^m [y^{(i)} log(\hat{y}^{(i)}) + (1 - y^{(i)}) log (1 - \hat{y}^{(i)})] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$$
# 
# #### Parameters Update
# 
# For updating the parameter, the usual rule is as follows, where $x^{(i)}$ is the i'th input data:
# 
# $$\theta_j = \theta_j - \alpha \; [ \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x^{(i)} + \frac{\lambda}{m} \theta_j]$$
# 
# However, due to homomorphic encryption constraint, we preferred to use an $\alpha = 1$ to reduce a multiplication and set $\frac{\lambda}{m} = 0.05$ which gets us to the following update rule:
# 
# $$\theta_j = \theta_j - [ \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x^{(i)} + 0.05 \theta_j]$$
# 
# #### Sigmoid Approximation
# 
# Since we can't simply compute sigmoid on encrypted data, we need to approximate it using a low degree polynomial, the lower the degree the better, as we aim to perform as few multiplications as possible, to be able to use smaller parameters and thus optimize computation. This tutorial uses a degree 3 polynomial from https://eprint.iacr.org/2018/462.pdf, which approximates the sigmoid function in the range $[-5,5]$.
# 
# $$\sigma(x) = 0.5 + 0.197 x - 0.004 x^3$$
# 
# #### Homomorphic Encryption Parameters
# 
# From the input data to the parameter update, a ciphertext will need a multiplicative depth of 6, 1 for the dot product operation, 2 for the sigmoid approximation, and 3 for the backpropagation phase (one is actually hidden in the `self._delta_w += enc_x * out_minus_y` operation in the `backward()` function, which is multiplying a 1-sized vector with an n-sized one, which requires masking the first slot and replicating it n times in the first vector). With a scale of around 20 bits, we need 6 coefficients modulus with the same bit-size as the scale, plus the last coefficient, which needs more bits, we are already out of the 4096 polynomial modulus degree (which requires < 109 total bit count of the coefficients modulus, if we consider 128-bit security), so we will use 8192. This will allow us to batch up to 4096 values in a single ciphertext, but we are far away from this limitation, so we shouldn't even think about it.
# 

 
class EncryptedLR:
    
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        # we accumulate gradients and counts the number of iterations
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out
    
    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1
        
    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        # update weights
        # We use a small regularization term to keep the output
        # of the linear layer in the range of the sigmoid approximation
        self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
        self.bias -= self._delta_b * (1 / self._count)
        # reset gradient accumulators and iterations count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
    
    @staticmethod
    def sigmoid(enc_x):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        # which fits the function pretty well in the range [-5,5]
        return enc_x.polyval([0.5, 0.197, 0, -0.004])
    
    def plain_accuracy(self, x_test, y_test):
        # evaluate accuracy of the model on
        # the plain (x_test, y_test) dataset
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        correct = torch.abs(y_test - out) < 0.5
        return correct.float().mean()    
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


 
# parameters
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
# create TenSEALContext
ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_training.global_scale = 2 ** 21
ctx_training.generate_galois_keys()

 
t_start = time()
enc_x_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in x_train]
enc_y_train = [ts.ckks_vector(ctx_training, y.tolist()) for y in y_train]
t_end = time()
print(f"Encryption of the training_set took {int(t_end - t_start)} seconds")

 
# Below we study the distribution of `x.dot(weight) + bias` in both plain and encrypted domains. Making sure that it falls into the range $[-5,5]$, which is where our sigmoid approximation is good at, and we don't want to feed it data that is out of this range so that we don't get erroneous output, which can make our training behave unpredictably. But the weights will change during the training process, and we should try to keep them as small as possible while still learning. A technique often used with logistic regression, and we do exactly this (but serving another purpose which is *generalization*), is known as *regularization*, and you might already have spotted the additional term `self.weight * 0.05` in the `update_parameters()` function, which is the result of doing regularization.
# 
# To recap, since our sigmoid approximation is only good in the range $[-5,5]$, we want to have all its inputs in that range. In order to do this, we need to keep our logistic regression parameters as small as possible, so we apply regularization.
# 
# **Note:** Keeping the parameters small certainly reduces the magnitude of the output, but we can also get out of range if the data wasn't standardized. You may have spotted that we standardized the data with a mean of 0 and std of 1, this was both for better performance, as well as to keep the inputs to the sigmoid in the desired range.

 
normal_dist = lambda x, mean, var: np.exp(- np.square(x - mean) / (2 * var)) / np.sqrt(2 * np.pi * var)

def plot_normal_dist(mean, var, rmin=-10, rmax=10):
    x = np.arange(rmin, rmax, 0.01)
    y = normal_dist(x, mean, var)
    fig = plt.plot(x, y)
    
# plain distribution
lr = LR(n_features)
data = lr.lr(x_test)
mean, var = map(float, [data.mean(), data.std() ** 2])
plot_normal_dist(mean, var)
print("Distribution on plain data:")
# plt.show()

# encrypted distribution
def encrypted_out_distribution(eelr, enc_x_test):
    w = eelr.weight
    b = eelr.bias
    data = []
    for enc_x in enc_x_test:
        enc_out = enc_x.dot(w) + b
        data.append(enc_out.decrypt())
    data = torch.tensor(data)
    mean, var = map(float, [data.mean(), data.std() ** 2])
    plot_normal_dist(mean, var)
    print("Distribution on encrypted data:")
    # plt.show()

eelr = EncryptedLR(lr)
eelr.encrypt(ctx_training)
encrypted_out_distribution(eelr, enc_x_train)


 
eelr = EncryptedLR(LR(n_features))
accuracy = eelr.plain_accuracy(x_test, y_test)
print(f"Accuracy at epoch #0 is {accuracy}")

times = []
for epoch in range(EPOCHS):
    eelr.encrypt(ctx_training)
    
    # if you want to keep an eye on the distribution to make sure
    # the function approximation is still working fine
    # WARNING: this operation is time consuming
    # encrypted_out_distribution(eelr, enc_x_train)
    
    t_start = time()
    for enc_x, enc_y in zip(enc_x_train, enc_y_train):
        enc_out = eelr.forward(enc_x)
        eelr.backward(enc_x, enc_out, enc_y)
    eelr.update_parameters()
    t_end = time()
    times.append(t_end - t_start)
    
    eelr.decrypt()
    accuracy = eelr.plain_accuracy(x_test, y_test)
    print(f"Accuracy at epoch #{epoch + 1} is {accuracy}")


print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
print(f"Final accuracy is {accuracy}")

diff_accuracy = plain_accuracy - accuracy
print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")
if diff_accuracy < 0:
    print("Oh! We got a better accuracy when training on encrypted data! The noise was on our side...")

# # Congratulations!!! - Time to Join the Community!
# 
# Congratulations on completing this notebook tutorial! If you enjoyed this and would like to join the movement towards privacy preserving, decentralized ownership of AI and the AI supply chain (data), you can do so in the following ways!
# 
# ### Star TenSEAL on GitHub
# 
# The easiest way to help our community is just by starring the Repos! This helps raise awareness of the cool tools we're building.
# 
# - [Star TenSEAL](https://github.com/OpenMined/TenSEAL)
# 
# ### Join our Slack!
# 
# The best way to keep up to date on the latest advancements is to join our community! You can do so by filling out the form at [http://slack.openmined.org](http://slack.openmined.org). #lib_tenseal and #code_tenseal are the main channels for the TenSEAL project.
# 
# ### Join our Team!
# 
# If you're excited about what we are working on TenSEAL, and if you're interested to work on homomorphic encryption related use cases, you should definitely join us!
# 
# [Apply to the crypto team!](https://docs.google.com/forms/d/1T6MJ21V1lb7aEr4ilZOTYQXzxXP6KbpLumZVmTZMSuY/edit)
# 
# 
# ### Donate
# 
# If you don't have time to contribute to our codebase, but would still like to lend support, you can also become a Backer on our Open Collective. All donations go toward our web hosting and other community expenses such as hackathons and meetups!
# 
# [OpenMined's Open Collective Page](https://opencollective.com/openmined)
# 


