import sys

sys.path.insert(1, "../")  
from sklearn.metrics import confusion_matrix, f1_score
import torch
import pandas as pd
import random
import numpy as np
np.random.seed(0)

# Initialize lists to store metrics over repetitions
accuracy_list = []
tpr_list = []
fpr_list = []
equalized_odds_list = []

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
    def adult_income_data():
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
        # print(data['female'])
        return split_train_test(x, y), data
        # after standardization, 'female' is -0.600922 for males and 1.663999 for females

    dataset_read_output = adult_income_data()
    x_train, y_train, x_test, y_test = dataset_read_output[0]
    adult_data = dataset_read_output[1]

    print("############# Data summary #############")
    print(f"x_train has shape: {x_train.shape}")
    print(f"y_train has shape: {y_train.shape}")
    print(f"x_test has shape: {x_test.shape}")
    print(f"y_test has shape: {y_test.shape}")
    print("#######################################")

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


    # define the number of epochs for plain training
    EPOCHS = 5

    def train(model, optim, criterion, x, y, epochs=EPOCHS):
        # print("model: ", model)
        # print("x shape:", x)
        # print("y shape:", y)
        for e in range(1, epochs + 1):
            optim.zero_grad()
            out = model(x)
            # print("out shape:", out)
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


    # Define a function to evaluate the model and display confusion matrix and F1 score
    def evaluate_model(model, x, y, privileged_groups_index):
        # Evaluate the model on the test set
        predictions = model(x)
        # print("predictions: ", predictions)
        
        # Convert predictions to binary values (0 or 1) using a threshold of 0.5
        binary_predictions = (predictions >= 0.5).float()
        # print("binary predictions: ", binary_predictions)
        
        # Convert ground truth to numpy array
        y_numpy = y.numpy()
        
        # Convert predictions to numpy array
        predictions_numpy = binary_predictions.detach().numpy()
        
        # Print debugging information
        print("Size of y_numpy:", y_numpy.shape)
        print("Max index in privileged_groups_index:", max(privileged_groups_index))

        # Identify predictions and ground truth for privileged and unprivileged groups
        y_privileged = y_numpy[privileged_groups_index]
        predictions_privileged = predictions_numpy[privileged_groups_index]

        # Calculate confusion matrix
        cm = confusion_matrix(y_privileged, predictions_privileged)
        
        # Calculate F1 score
        f1 = f1_score(y_privileged, predictions_privileged)
        
        # Display confusion matrix and F1 score
        print("Confusion Matrix:")
        print(cm)
        print("\nF1 Score:", f1)
        return cm

    # Evaluate the model on the test set and display results
    privileged_indices = []

    for index, row in adult_data.iterrows():
        # since the values are floating point numbers, it might be difficult to equate the exact value to -0.600922 so I am using inequality
        if row["female"] < 0:
            # print(row["female"])
            privileged_indices.append(index)

    privileged_group_indices = privileged_indices[1]
    privileged_group_indices = [index for index, row in adult_data.iterrows() if row['female'] < 0]
    privileged_group_indices = [index[1] for index in privileged_indices if index[1] < 4504]

    # print("privileged_group_indices: ", privileged_group_indices)
    conf_matrix = evaluate_model(model, x_test, y_test, privileged_group_indices)

    def calculate_metrics_from_confusion_matrix(confusion_matrix):
        # Extract values from the confusion matrix
        tn_protected, fp_protected, fn_protected, tp_protected = confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0], confusion_matrix[1, 1]

        tpr_protected = tp_protected / (tp_protected + fn_protected) if (tp_protected + fn_protected) > 0 else 0
        fpr_protected = fp_protected / (fp_protected + tn_protected) if (fp_protected + tn_protected) > 0 else 0
        tnr_protected = tn_protected / (tn_protected + fp_protected) if (tn_protected + fp_protected) > 0 else 0
        fnr_protected = fn_protected / (fn_protected + tp_protected) if (fn_protected + tp_protected) > 0 else 0

        equalized_odds_protected = tpr_protected - fpr_protected 
        return tpr_protected, fpr_protected, equalized_odds_protected

    tpr_protected, fpr_protected, equalized_odds_protected = calculate_metrics_from_confusion_matrix(confusion_matrix=conf_matrix)
    print(f"True Positive Rate (TPR) for protected group: {tpr_protected}")
    print(f"False Positive Rate (FPR) for protected group: {fpr_protected}")
    print(f"Equalized Odds for protected group: {equalized_odds_protected}")

    accuracy_list.append(plain_accuracy)
    tpr_list.append(tpr_protected)
    fpr_list.append(fpr_protected)
    equalized_odds_list.append(equalized_odds_protected)


    with open("PlainAdultCorrected.txt", "a") as file:
        file.write(f"Repetition {repetition + 1}:\n")
        file.write(f"Accuracy: {plain_accuracy}\n")
        file.write(f"TPR for protected group: {tpr_protected}\n")
        file.write(f"FPR for protected group: {fpr_protected}\n")
        file.write(f"Equalized Odds for protected group: {equalized_odds_protected}\n")
        file.write(f"Confusion Matrix:\n{conf_matrix}\n\n")

# Calculate and append average metrics to the output file
with open("PlainAdultCorrected.txt", "a") as file:
    file.write("\nAverage Metrics:\n")
    file.write(f"Average Accuracy: {np.mean(accuracy_list)}\n")
    file.write(f"Average TPR for protected group: {np.mean(tpr_list)}\n")
    file.write(f"Average FPR for protected group: {np.mean(fpr_list)}\n")
    file.write(f"Average Equalized Odds for protected group: {np.mean(equalized_odds_list)}\n")
