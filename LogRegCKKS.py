import torch
import tenseal as ts
import pandas as pd
import random
from time import time
from sklearn.metrics import confusion_matrix, f1_score

# those are optional and are not necessary for training
import numpy as np
import matplotlib.pyplot as plt

# torch.random.manual_seed(73)
# random.seed(73)


# Initialize lists to store metrics over repetitions
plain_accuracy_list = []
encrypted_accuracy_list = []
diff_accuracy_list = []
tpr_list = []
fpr_list = []
equalized_odds_list = []

for repetition in range(10):


    def split_train_test(x, y, test_ratio=0.3):
        idxs = [i for i in range(len(x))]
        random.shuffle(idxs)
        # delimiter between test and train data
        delim = int(len(x) * test_ratio)
        test_idxs, train_idxs = idxs[:delim], idxs[delim:]
        return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]


    def heart_disease_data():
        data = pd.read_csv("./adultPreprocessed.csv")
        # drop rows with missing values
        data = data.dropna()
        # balance data
        grouped = data.groupby('income')
        data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
        # extract labels
        y = torch.tensor(data["income"].values).float().unsqueeze(1)
        data = data.drop("income", axis='columns')
        # standardize data
        data = (data - data.mean()) / data.std()
        x = torch.tensor(data.values).float()
        return split_train_test(x, y)


    def adult_income_data():
        pass


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
    features_np = x_train.numpy()
    model = LR(n_features)
    label = y_train.numpy()
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

    plt.close('all')

    enc_features = [ts.ckks_vector(ctx_eval, f.tolist()) for f in features_np]

    class CustomBinaryLabelDataset:
        def __init__(self, features, labels, biased_column_index=0, feature_shape=0):
            self.features = features
            self.labels = labels
            self.biased_column_index = biased_column_index
            self.instance_count = len(features)
            self.feature_shape = feature_shape

        def get_biased_column(self):
            decrypted_features = [feature.decrypt() for feature in self.features]
            return [decrypted_feature[self.biased_column_index] for decrypted_feature in decrypted_features]
            # return [feature[self.biased_column_index] for feature in self.features]

    # Assuming biased column is female (column index 0), you can create the dataset as follows:
    biased_column_index = 3
    custom_dataset = CustomBinaryLabelDataset(features=enc_features, labels=label, biased_column_index=biased_column_index, feature_shape=x_test.shape)
    # print(custom_dataset)
    # print(type(custom_dataset))


    def get_predictions(model, enc_x_test):
        predictions = []
        for enc_x in enc_x_test:
            enc_out = model(enc_x)
            out = enc_out.decrypt()
            out = torch.tensor(out)
            out = torch.sigmoid(out)
            predictions.append(out.item())
        return predictions

    def convert_predictions(predictions, threshold=0.5):
        # Assuming threshold of 0.5, adjust as needed
        binary_predictions = [1 if pred >= threshold else 0 for pred in predictions]
        return binary_predictions

    def compute_f1_score(true_labels, binary_predictions):
        f1 = f1_score(true_labels, binary_predictions)
        return f1

    predictions = get_predictions(eelr, enc_x_test)
    binary_predictions = convert_predictions(predictions)
    true_labels = y_test.numpy().flatten()

    conf_matrix = confusion_matrix(true_labels, binary_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute F1 Score
    f1 = compute_f1_score(true_labels, binary_predictions)
    print(f"F1 Score: {f1}")

    def calculate_disparate_impact(predictions, protected_group_indices):
        # Assuming 1 indicates positive prediction
        favorable_protected = sum(predictions[protected_group_indices] == 1)
        favorable_unprotected = sum(predictions[~protected_group_indices] == 1)
        disparate_impact = favorable_protected / favorable_unprotected
        return disparate_impact

    def calculate_metrics_from_confusion_matrix(confusion_matrix):
        # Extract values from the confusion matrix
        tn_protected, fp_protected, fn_protected, tp_protected = confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0], confusion_matrix[1, 1]

        tpr_protected = tp_protected / (tp_protected + fn_protected) if (tp_protected + fn_protected) > 0 else 0
        fpr_protected = fp_protected / (fp_protected + tn_protected) if (fp_protected + tn_protected) > 0 else 0

        tnr_protected = tn_protected / (tn_protected + fp_protected) if (tn_protected + fp_protected) > 0 else 0
        equalized_odds_protected = tpr_protected / (1 - tnr_protected) if (1 - tnr_protected) > 0 else 0

        return tpr_protected, fpr_protected, equalized_odds_protected

    def calculate_demographic_parity(predictions, protected_group_indices):
        positive_rate_protected = sum(predictions[protected_group_indices] == 1) / len(protected_group_indices)
        positive_rate_unprotected = sum(predictions[~protected_group_indices] == 1) / len(~protected_group_indices)

        return positive_rate_protected, positive_rate_unprotected

    # Get Predictions
    # predictions = get_predictions(eelr, enc_x_test)
    tpr_protected, fpr_protected, equalized_odds_protected = calculate_metrics_from_confusion_matrix(confusion_matrix=conf_matrix)
    print(f"True Positive Rate (TPR) for protected group: {tpr_protected}")
    print(f"False Positive Rate (FPR) for protected group: {fpr_protected}")
    print(f"Equalized Odds for protected group: {equalized_odds_protected}")
    # demographic_parity = calculate_demographic_parity(predictions, custom_dataset.get_biased_column())
    # print(f"Demographic Parity: {demographic_parity}")
    # disparate_impact = calculate_disparate_impact(predictions, custom_dataset.get_biased_column())
    # print(f"Disparate Impact: {disparate_impact}")
    # accuracy_list.append(plain_accuracy)
    plain_accuracy_list.append(plain_accuracy)
    encrypted_accuracy_list.append(encrypted_accuracy)
    diff_accuracy_list.append(diff_accuracy)
    tpr_list.append(tpr_protected)
    fpr_list.append(fpr_protected)
    equalized_odds_list.append(equalized_odds_protected)


    with open("CKKS.txt", "a") as file:
        file.write(f"Repetition {repetition + 1}:\n")
        file.write(f"Plain Accuracy: {plain_accuracy}\n")
        file.write(f"Encrypet Accuracy: {encrypted_accuracy}\n")
        file.write(f"Difference between plain and encrypted accuracies: {diff_accuracy}\n")
        file.write(f"TPR for protected group: {tpr_protected}\n")
        file.write(f"FPR for protected group: {fpr_protected}\n")
        file.write(f"Equalized Odds for protected group: {equalized_odds_protected}\n\n")
        file.write(f"Confusion Matrix:\n{conf_matrix}")

with open("CKKS.txt", "a") as file:
    file.write("Average Metrics:\n")
    file.write(f"Average Plain Accuracy: {np.mean(plain_accuracy_list)}\n")
    file.write(f"Average Encrypted Accuracy: {np.mean(encrypted_accuracy_list)}\n")
    file.write(f"Average Difference between plain and encrypted accuracies: {np.mean(diff_accuracy_list)}\n")
    file.write(f"Average TPR for protected group: {np.mean(tpr_list)}\n")
    file.write(f"Average FPR for protected group: {np.mean(fpr_list)}\n")
    file.write(f"Average Equalized Odds for protected group: {np.mean(equalized_odds_list)}\n")
