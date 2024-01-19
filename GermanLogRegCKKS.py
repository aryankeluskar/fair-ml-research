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
tpr_list = []
fpr_list = []
equalized_odds_list = []
predictive_rate_parity_list = []
demographic_parity_list = []
proportional_parity_list = []
accuracy_parity_list = []
matthews_correlation_coefficient_list = []

# Define the number of repetitions for the experiment here
ITERATIONS = 10

for repetition in range(ITERATIONS):
    # separate train and test data, alloting 30% of the data for testing
    def split_train_test(x, y, test_ratio=0.3):
        idxs = [i for i in range(len(x))]
        random.shuffle(idxs)
        # delimiter between test and train data
        delim = int(len(x) * test_ratio)
        test_idxs, train_idxs = idxs[:delim], idxs[delim:]
        return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

    # read adult income data, standardize it, and split into train and test sets
    def german_credit_data():
        data = pd.read_csv("./germanPreprocessed.csv")
        # drop rows with missing values
        data = data.dropna()
        # balance data
        grouped = data.groupby('Creditability')
        data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
        # extract labels
        y = torch.tensor(data["Creditability"].values).float().unsqueeze(1)
        data = data.drop("Creditability", axis='columns')
        # standardize data
        data = (data - data.mean()) / data.std()
        x = torch.tensor(data.values).float()
        return split_train_test(x, y), data

    dataset_read_output = german_credit_data()
    x_train, y_train, x_test, y_test = dataset_read_output[0]
    # print(f"type and x_test: {type(x_test)}{x_test}")
    # print(f"type and y_test: {type(y_test)}{y_test}")

    x_test_df = pd.DataFrame(x_test.numpy())
    for index, row in x_test_df.iterrows():
        if row[19] > 0:
            print(row[19])

    adult_data = dataset_read_output[1]

    print("############# Data summary #############")
    print(f"x_train has shape: {x_train.shape}")
    print(f"y_train has shape: {y_train.shape}")
    print(f"x_test has shape: {x_test.shape}")
    print(f"y_test has shape: {y_test.shape}")
    print("#######################################")

    # Plain Logistic Regression for testing purposes
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

    # Old Version which used AIF360
    # privileged_groups = [{'female': 0}]
    # unprivileged_groups = [{'female': 1}]

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


    t_start = time()
    enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
    t_end = time()
    print(f"Encryption of the test-set took {int(t_end - t_start)} seconds")


    # (optional) encrypt the model's parameters
    # eelr.encrypt(ctx_eval)


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

    # CustomBinaryLabelDataset is not in the tenseal library, so I created it here
    # the normal BinaryLabelDataset class in tenseal does not allow for CKKS vectors
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

    # biased column is foreign worker (column index 20 in germanPreprocessed.csv), you can create the dataset as follows:
    biased_column_index = 20
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

    privileged_indices = []

    for index, row in y_test.iterrows():
        # since the values are floating point numbers, it might be difficult to equate the exact value to -0.600922 so I am using inequality
        if row["Foreign Worker"] > 0:
            # print(row["female"])
            privileged_indices.append(index)

    privileged_group_indices = privileged_indices[1]
    privileged_group_indices = [index for index, row in y_test.iterrows() if row['Foreign Worker'] > 0]
    privileged_group_indices = [index[1] for index in privileged_indices if index[1] < 4504]

    # Evaluate the model on the test set
    predictions_list = get_predictions(eelr, enc_x_test)
    predictions = torch.tensor(predictions_list).unsqueeze(1)

    # print("predictions: ", predictions)

    # Convert predictions to binary values (0 or 1) using a threshold of 0.5
    binary_predictions = (predictions >= 0.5).float()
    # print("binary predictions: ", binary_predictions)
    
    # Convert ground truth to numpy array
    y_numpy = y_test.numpy()
    
    # Convert predictions to numpy array
    predictions_numpy = binary_predictions.detach().numpy()
    
    # Print debugging information
    # print("Size of y_numpy:", y_numpy.shape)
    # print("Max index in privileged_group_indices:", max(privileged_group_indices))

    # Identify predictions and ground truth for privileged and unprivileged groups
    y_privileged = y_numpy[privileged_group_indices]
    predictions_privileged = predictions_numpy[privileged_group_indices]

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_privileged, predictions_privileged)
    
    # Calculate F1 score
    f1 = f1_score(y_privileged, predictions_privileged)
        

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
        fnr_protected = fn_protected / (fn_protected + tp_protected) if (fn_protected + tp_protected) > 0 else 0

        predictive_rate_parity = tp_protected / (tp_protected+fp_protected)
        demographic_parity = (tp_protected+fp_protected) 
        proportional_parity = (tp_protected+fp_protected)/(tp_protected+fn_protected+fp_protected+tn_protected) 
        accuracy_parity = (tp_protected+tn_protected)/(tp_protected+fn_protected+fp_protected+tn_protected)
        matthews_correlation_coefficient = (tp_protected*tn_protected-fp_protected*fn_protected)/np.sqrt((tp_protected+fp_protected)*(tp_protected+fn_protected)*(tn_protected+fp_protected)*(tn_protected+fn_protected))
        equalized_odds_protected = tpr_protected / (tpr_protected+fn_protected) 
        
        return tpr_protected, fpr_protected, equalized_odds_protected, predictive_rate_parity, demographic_parity, proportional_parity, accuracy_parity, matthews_correlation_coefficient

    # Get Predictions
    # predictions = get_predictions(eelr, enc_x_test)
    tpr_protected, fpr_protected, equalized_odds_protected, predictive_rate_parity, demographic_parity, proportional_parity, accuracy_parity, matthews_correlation_coefficient = calculate_metrics_from_confusion_matrix(confusion_matrix=conf_matrix)
    print(f"True Positive Rate (TPR) for protected group: {tpr_protected}")
    print(f"False Positive Rate (FPR) for protected group: {fpr_protected}")
    print(f"Equalized Odds for protected group: {equalized_odds_protected}")
    print(f"Predictive Rate Parity for protected group: {predictive_rate_parity}")
    print(f"Demographic Parity for protected group: {demographic_parity}")
    print(f"Proportional Parity for protected group: {proportional_parity}")
    print(f"Accuracy Parity for protected group: {accuracy_parity}")
    print(f"Matthews Correlation Coefficient for protected group: {matthews_correlation_coefficient}")

    plain_accuracy_list.append(plain_accuracy)
    encrypted_accuracy_list.append(encrypted_accuracy)
    diff_accuracy_list.append(diff_accuracy)
    tpr_list.append(tpr_protected)
    fpr_list.append(fpr_protected)
    equalized_odds_list.append(equalized_odds_protected)
    predictive_rate_parity_list.append(predictive_rate_parity)
    demographic_parity_list.append(demographic_parity)
    proportional_parity_list.append(proportional_parity)
    accuracy_parity_list.append(accuracy_parity)
    matthews_correlation_coefficient_list.append(matthews_correlation_coefficient)

    print("THIS IS RUN NUMBER: ", repetition + 1)

    with open("GermanAdultMoreMetrics.txt", "a") as file:
        file.write(f"Repetition {repetition + 1}:\n")
        file.write(f"Plain Accuracy: {plain_accuracy}\n")
        file.write(f"Encrypted Accuracy: {encrypted_accuracy}\n")
        file.write(f"Difference between plain and encrypted accuracies: {diff_accuracy}\n")
        file.write(f"TPR for protected group: {tpr_protected}\n")
        file.write(f"FPR for protected group: {fpr_protected}\n")
        file.write(f"Equalized Odds for protected group: {equalized_odds_protected}\n")
        file.write(f"Predictive Rate Parity for protected group: {predictive_rate_parity}\n")
        file.write(f"Demographic Parity for protected group: {demographic_parity}\n")
        file.write(f"Proportional Parity for protected group: {proportional_parity}\n")
        file.write(f"Accuracy Parity for protected group: {accuracy_parity}\n")
        file.write(f"Matthews Correlation Coefficient for protected group: {matthews_correlation_coefficient}\n")
        file.write(f"Confusion Matrix:\n{conf_matrix}\n\n")

        # In case there are a lot of ITERATIONS and you want to play it safe by saving average metrics every 5 runs 
        # if repetition % 5 == 0:
        #     file.write(f"\nAverage Metrics till run number:{repetition+1}\n")
        #     file.write(f"Average Plain Accuracy: {np.mean(plain_accuracy_list)}\n")
        #     file.write(f"Average Encrypted Accuracy: {np.mean(encrypted_accuracy_list)}\n")
        #     file.write(f"Average Difference between plain and encrypted accuracies: {np.mean(diff_accuracy_list)}\n")
        #     file.write(f"Average TPR for protected group: {np.mean(tpr_list)}\n")
        #     file.write(f"Average FPR for protected group: {np.mean(fpr_list)}\n")
        #     file.write(f"Average Equalized Odds for protected group: {np.mean(equalized_odds_list)}\n") 

with open("GermanAdultMoreMetrics.txt", "a") as file:
    file.write("\nAverage Metrics:\n")
    file.write(f"Average Plain Accuracy: {np.mean(plain_accuracy_list)}\n")
    file.write(f"Average Encrypted Accuracy: {np.mean(encrypted_accuracy_list)}\n")
    file.write(f"Average Difference between plain and encrypted accuracies: {np.mean(diff_accuracy_list)}\n")
    file.write(f"Average TPR for protected group: {np.mean(tpr_list)}\n")
    file.write(f"Average FPR for protected group: {np.mean(fpr_list)}\n")
    file.write(f"Average Equalized Odds for protected group: {np.mean(equalized_odds_list)}\n")
    file.write(f"Average Predictive Rate Parity for protected group: {np.mean(predictive_rate_parity_list)}\n")
    file.write(f"Average Demographic Parity for protected group: {np.mean(demographic_parity_list)}\n")
    file.write(f"Average Proportional Parity for protected group: {np.mean(proportional_parity_list)}\n")
    file.write(f"Average Accuracy Parity for protected group: {np.mean(accuracy_parity_list)}\n")
    file.write(f"Average Matthews Correlation Coefficient for protected group: {np.mean(matthews_correlation_coefficient_list)}\n")

