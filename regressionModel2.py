import random
import numpy as np
import csv
import pickle

tuned_params_count = {
    'n_trees': 0,
    'max_depth': 0,
    'min_size': 0,
    'thresholds': 0,
    'features': 0
}

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
        headers = data[0]  
        dataset = [[float(value) for value in row] for row in data[1:]]
    return headers, dataset

def train_test_split(dataset, test_size=0.2):
    test_indices = random.sample(range(len(dataset)), int(test_size * len(dataset)))
    train_set = [row for idx, row in enumerate(dataset) if idx not in test_indices]
    test_set = [row for idx, row in enumerate(dataset) if idx in test_indices]
    return train_set, test_set

def split_dataset(dataset, feature_idx, threshold):
    left_split = [row for row in dataset if row[feature_idx] <= threshold]
    right_split = [row for row in dataset if row[feature_idx] > threshold]
    return left_split, right_split

def calculate_mse(groups, output_idx):
    mse = 0
    for group in groups:
        if not group:
            continue
        outputs = [row[output_idx] for row in group]
        mean_output = np.mean(outputs)
        mse += sum((y - mean_output) ** 2 for y in outputs)
    return mse

def get_best_split(dataset, output_idx, n_features):
    best_split = {}
    min_mse = float("inf")
    
    feature_indices = list(range(len(dataset[0])))  
    feature_indices.remove(output_idx)  
    selected_features = random.sample(feature_indices, n_features) 

    for feature_idx in selected_features:
        thresholds = set(row[feature_idx] for row in dataset)
        for threshold in thresholds:
            groups = split_dataset(dataset, feature_idx, threshold)
            mse = calculate_mse(groups, output_idx)
            if mse < min_mse:
                min_mse = mse
                best_split = {
                    'feature_idx': feature_idx,
                    'threshold': threshold,
                    'groups': groups
                }
    tuned_params_count['thresholds'] += 1
    tuned_params_count['features'] += n_features 
    return best_split

def create_leaf(group, output_idx):
    outputs = [row[output_idx] for row in group]
    return np.mean(outputs)

def build_tree(node, max_depth, min_size, depth, output_idx):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = create_leaf(left + right, output_idx)
        return

    if depth >= max_depth:
        node['left'], node['right'] = create_leaf(left, output_idx), create_leaf(right, output_idx)
        return

    if len(left) <= min_size:
        node['left'] = create_leaf(left, output_idx)
    else:
        node['left'] = get_best_split(left, output_idx,n_features)
        build_tree(node['left'], max_depth, min_size, depth + 1, output_idx)

    if len(right) <= min_size:
        node['right'] = create_leaf(right, output_idx)
    else:
        node['right'] = get_best_split(right, output_idx,n_features)
        build_tree(node['right'], max_depth, min_size, depth + 1, output_idx)

def predict_with_tree(tree, row):
    if isinstance(tree, dict):
        if row[tree['feature_idx']] <= tree['threshold']:
            return predict_with_tree(tree['left'], row)
        else:
            return predict_with_tree(tree['right'], row)
    else:
        return tree


def bootstrap_sample(dataset):
    n_samples = len(dataset)
    return [dataset[random.randint(0, n_samples - 1)] for _ in range(n_samples)]

def build_random_forest(train_set, n_trees, max_depth, min_size, output_idx, n_features):
    forest = []
    for _ in range(n_trees):
        tuned_params_count['n_trees'] += 1
        
        sample = bootstrap_sample(train_set)
        tree = get_best_split(sample, output_idx, n_features)
        build_tree(tree, max_depth, min_size, 1, output_idx)
        forest.append(tree)
        
        tuned_params_count['max_depth'] += 1
        tuned_params_count['min_size'] += 1
        
    return forest


def predict_with_forest(forest, row):
    predictions = [predict_with_tree(tree, row) for tree in forest]
    return np.mean(predictions)


def evaluate_model(y_true, y_pred):
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    r2 = 1 - (sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / 
              sum((y_true[i] - np.mean(y_true)) ** 2 for i in range(len(y_true))))
    return mse, r2

def calculate_variance(y_true):
    return np.var(y_true)

def normalize_mse_percentage(mse, variance):
    return (mse / variance) * 100


file_path = "Regression_Dataset.csv"  
headers, dataset = read_dataset(file_path)

train_set, test_set = train_test_split(dataset, test_size=0.2)
output_idx = len(dataset[0]) - 1  

n_trees = 50
max_depth = 10  
min_size = 5
n_features = int(np.sqrt(len(headers) - 1))  

forest = build_random_forest(train_set, n_trees, max_depth, min_size, output_idx, n_features)



y_true = [row[output_idx] for row in test_set]

y_pred = [predict_with_forest(forest, row) for row in test_set]



mse, r2 = evaluate_model(y_true, y_pred)
variance = calculate_variance(y_true)
normalized_mse_percentage = normalize_mse_percentage(mse, variance)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

def calculate_accuracy(y_true, y_pred):
    mape = calculate_mape(y_true, y_pred)
    return 100 - mape

mape = calculate_mape(y_true, y_pred)
accuracy = calculate_accuracy(y_true, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print(f"Normalized MSE as percentage: {normalized_mse_percentage:.2f}%")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

# Save the model to a file
with open('random_forest_regression_model.pkl', 'wb') as f:
    pickle.dump(forest, f)
print("Model saved successfully!")

# Print the tuned parameters count
print("\nTuned Parameters Count:")
for param, count in tuned_params_count.items():
    print(f"{param}: {count}")
