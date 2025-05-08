"""
Project 4: Support Vector Machines (SVMs) for MNIST Classification
This notebook implements SVM classification on MNIST data with different kernels and hyperparameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import time
from IPython.display import display

# Set random seed for reproducibility
np.random.seed(42)

# ----- 1. Data Preparation -----
print("Fetching MNIST dataset...")
# Fetch MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Scale the data (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Shuffle and split into 10k training and 10k testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=10000, test_size=10000, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")

# ----- 2. Visualize some examples from the dataset -----
def plot_digits(X, y, indices):
    """Plot some example digits for visualization"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Sample 10 random images from training set
random_indices = np.random.choice(len(X_train), 10, replace=False)
plot_digits(X_train, y_train, random_indices)

# ----- 3. Cross-validation with different kernels and hyperparameters -----
print("Running cross-validation with different kernels and hyperparameters...")

# Define parameter combinations to test
param_grid = {
    'linear': {'C': [0.1, 1, 10]},
    'poly': {
        'C': [0.1, 1, 10],
        'degree': [2, 3]
    },
    'rbf': {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 'scale']
    }
}

# Initialize results dictionary
results = {
    'kernel': [],
    'parameters': [],
    'mean_accuracy': [],
    'std_accuracy': [],
    'training_time': []
}

# Run 3-fold cross-validation for each kernel and parameter combination
for kernel_type in param_grid:
    for C in param_grid[kernel_type]['C']:
        if kernel_type == 'linear':
            params = {'C': C}
            model = SVC(kernel=kernel_type, C=C, random_state=42)
            param_str = f"C={C}"
        
        elif kernel_type == 'poly':
            for degree in param_grid[kernel_type]['degree']:
                params = {'C': C, 'degree': degree}
                model = SVC(kernel=kernel_type, C=C, degree=degree, random_state=42)
                param_str = f"C={C}, degree={degree}"
                
                start_time = time.time()
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                end_time = time.time()
                
                # Store results
                results['kernel'].append(kernel_type)
                results['parameters'].append(param_str)
                results['mean_accuracy'].append(cv_scores.mean())
                results['std_accuracy'].append(cv_scores.std())
                results['training_time'].append(end_time - start_time)
                
                print(f"Kernel: {kernel_type}, {param_str}")
                print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"Time: {end_time - start_time:.2f} seconds\n")
                
            continue  # Skip the code below for poly kernel
                
        elif kernel_type == 'rbf':
            for gamma in param_grid[kernel_type]['gamma']:
                params = {'C': C, 'gamma': gamma}
                model = SVC(kernel=kernel_type, C=C, gamma=gamma, random_state=42)
                param_str = f"C={C}, gamma={gamma}"
                
                start_time = time.time()
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                end_time = time.time()
                
                # Store results
                results['kernel'].append(kernel_type)
                results['parameters'].append(param_str)
                results['mean_accuracy'].append(cv_scores.mean())
                results['std_accuracy'].append(cv_scores.std())
                results['training_time'].append(end_time - start_time)
                
                print(f"Kernel: {kernel_type}, {param_str}")
                print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"Time: {end_time - start_time:.2f} seconds\n")
                
            continue  # Skip the code below for rbf kernel
        
        # For linear kernel
        start_time = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        end_time = time.time()
        
        # Store results
        results['kernel'].append(kernel_type)
        results['parameters'].append(param_str)
        results['mean_accuracy'].append(cv_scores.mean())
        results['std_accuracy'].append(cv_scores.std())
        results['training_time'].append(end_time - start_time)
        
        print(f"Kernel: {kernel_type}, {param_str}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Time: {end_time - start_time:.2f} seconds\n")

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# Format mean and std accuracy as percentage
results_df['accuracy'] = results_df.apply(
    lambda row: f"{row['mean_accuracy']*100:.2f}% ± {row['std_accuracy']*100:.2f}%", 
    axis=1
)

# Display results table
print("\nCross-validation Results:")
display(results_df[['kernel', 'parameters', 'accuracy', 'training_time']])

# ----- 4. Select best model based on CV accuracy -----
best_idx = results_df['mean_accuracy'].argmax()
best_kernel = results_df.loc[best_idx, 'kernel']
best_params = results_df.loc[best_idx, 'parameters']
best_accuracy = results_df.loc[best_idx, 'mean_accuracy']

print(f"\nBest model from cross-validation:")
print(f"Kernel: {best_kernel}, Parameters: {best_params}")
print(f"CV Accuracy: {best_accuracy*100:.2f}%")

# ----- 5. Train best model on full training set and evaluate on test set -----
print("\nTraining best model on full training set...")

# Parse the best parameters string
best_params_dict = {}
if best_kernel == 'linear':
    C_value = float(best_params.split('=')[1])
    best_model = SVC(kernel=best_kernel, C=C_value, random_state=42)
elif best_kernel == 'poly':
    params = best_params.split(', ')
    C_value = float(params[0].split('=')[1])
    degree_value = int(params[1].split('=')[1])
    best_model = SVC(kernel=best_kernel, C=C_value, degree=degree_value, random_state=42)
elif best_kernel == 'rbf':
    params = best_params.split(', ')
    C_value = float(params[0].split('=')[1])
    gamma_value = params[1].split('=')[1]
    # Convert gamma to float if it's not 'scale'
    if gamma_value != 'scale':
        gamma_value = float(gamma_value)
    best_model = SVC(kernel=best_kernel, C=C_value, gamma=gamma_value, random_state=42)

# Train the best model
start_time = time.time()
best_model.fit(X_train, y_train)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----- 6. Visualize confusion matrix -----
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ----- 7. Visualize some predictions -----
def plot_predictions(X, y_true, y_pred, indices):
    """Plot some predictions with true and predicted labels"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        correct = y_true[idx] == y_pred[idx]
        color = 'green' if correct else 'red'
        axes[i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}", color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Sample some random test examples
random_test_indices = np.random.choice(len(X_test), 10, replace=False)
plot_predictions(X_test, y_test, y_pred, random_test_indices)

# ----- 8. Plot examples of misclassified digits -----
# Find misclassified examples
misclassified = np.where(y_test != y_pred)[0]

if len(misclassified) > 0:
    print(f"Found {len(misclassified)} misclassified examples")
    
    # Plot some misclassified examples
    sample_size = min(10, len(misclassified))
    random_misclassified = np.random.choice(misclassified, sample_size, replace=False)
    
    plot_predictions(X_test, y_test, y_pred, random_misclassified)
else:
    print("No misclassified examples found!")

# ----- 9. Effect of hyperparameters visualization -----
# Visualize effect of C parameter for different kernels
plt.figure(figsize=(12, 8))
for kernel in ['linear', 'poly', 'rbf']:
    kernel_results = results_df[results_df['kernel'] == kernel]
    
    if kernel == 'linear':
        # Extract C values and corresponding accuracies
        C_values = [float(param.split('=')[1]) for param in kernel_results['parameters']]
        plt.plot(C_values, kernel_results['mean_accuracy'], 'o-', label=f'{kernel} kernel')
    
    elif kernel == 'rbf':
        # Group by C value for the scale gamma
        scale_results = kernel_results[kernel_results['parameters'].str.contains('gamma=scale')]
        if not scale_results.empty:
            C_values = [float(param.split('=')[1].split(',')[0]) for param in scale_results['parameters']]
            plt.plot(C_values, scale_results['mean_accuracy'], 's-', label=f'{kernel} kernel (gamma=scale)')
        
        # Group by C value for gamma=0.1
        gamma_results = kernel_results[kernel_results['parameters'].str.contains('gamma=0.1')]
        if not gamma_results.empty:
            C_values = [float(param.split('=')[1].split(',')[0]) for param in gamma_results['parameters']]
            plt.plot(C_values, gamma_results['mean_accuracy'], '^-', label=f'{kernel} kernel (gamma=0.1)')
    
    elif kernel == 'poly':
        # Group by C value for degree=2
        degree2_results = kernel_results[kernel_results['parameters'].str.contains('degree=2')]
        if not degree2_results.empty:
            C_values = [float(param.split('=')[1].split(',')[0]) for param in degree2_results['parameters']]
            plt.plot(C_values, degree2_results['mean_accuracy'], 'D-', label=f'{kernel} kernel (degree=2)')
        
        # Group by C value for degree=3
        degree3_results = kernel_results[kernel_results['parameters'].str.contains('degree=3')]
        if not degree3_results.empty:
            C_values = [float(param.split('=')[1].split(',')[0]) for param in degree3_results['parameters']]
            plt.plot(C_values, degree3_results['mean_accuracy'], 'X-', label=f'{kernel} kernel (degree=3)')

plt.xscale('log')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Cross-validation Accuracy')
plt.title('Effect of Regularization Parameter C on Different Kernels')
plt.legend()
plt.grid(True)
plt.show()

# ----- 10. Summary and Conclusions -----
print("\n----- Summary -----")
print(f"Best Model Configuration: {best_kernel} kernel with {best_params}")
print(f"Cross-validation Accuracy: {best_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print("\nInsights:")
print("1. The regularization parameter C controls the trade-off between maximizing the margin and minimizing the classification error.")
print("2. Different kernels transform the feature space in different ways, affecting how the decision boundary is drawn.")
print("3. The RBF kernel typically performs well on image data like MNIST because it can capture complex, non-linear relationships.")
print("4. Polynomial kernels with higher degrees can capture more complex decision boundaries but may overfit.")
print("5. Linear kernels are faster but may not capture the complexity of the data as well as non-linear kernels.")