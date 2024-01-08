import torch
import tenseal as ts
from aif360.datasets import BinaryLabelDataset

# Step 1: Prepare your data (features, labels, sensitive attributes)
# Replace the following with your actual data
features = torch.randn(100, 10)  # Example: 100 samples, 10 features
labels = torch.randint(2, size=(100, 1)).float()  # Binary labels (0 or 1)
sensitive_attributes = torch.randint(2, size=(100, 1)).float()  # Binary sensitive attribute (0 or 1)

# Step 2: Convert PyTorch tensors to NumPy arrays
features_np = features.numpy()
labels_np = labels.numpy()
sensitive_attributes_np = sensitive_attributes.numpy()

poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
context.global_scale = 2 ** 20
context.generate_galois_keys()

enc_features = [ts.ckks_vector(context, f.tolist()) for f in features_np]

# Step 4: Create BinaryLabelDataset
dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                             features=enc_features, labels=labels_np,
                             protected_attributes=sensitive_attributes_np)

# Step 5: Print dataset information
print("Number of instances: {}".format(dataset.instance_count))
print("Number of features: {}".format(dataset.feature_shape))
print("Favorable label: {}".format(dataset.favorable_label))
print("Unfavorable label: {}".format(dataset.unfavorable_label))
print("Protected attribute: {}".format(dataset.protected_attribute))

# Optionally, you can perform fairness metric calculations using AIF360
from aif360.metrics import BinaryLabelDatasetMetric

metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[...], privileged_groups=[...])
print("Difference in mean outcomes between unprivileged and privileged groups: {}".format(metric.mean_difference()))
