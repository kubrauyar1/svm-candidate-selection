import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate synthetic data
n_samples = 200
experience_years = np.random.uniform(0, 10, n_samples)
technical_score = np.random.uniform(0, 100, n_samples)

# Create labels based on the given criteria
labels = []
for experience, score in zip(experience_years, technical_score):
    if experience < 2 and score < 60:
        labels.append(1)  # Not hired
    else:
        labels.append(0)  # Hired

# Prepare the data
X = np.column_stack((experience_years, technical_score))
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title="Job Application Evaluation: SVM Decision Boundary", filename=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=60, edgecolors='k', alpha=0.7)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=150, linewidth=1.5, facecolors='none', edgecolors='k')

    plt.title(title)
    plt.xlabel("Experience Years (standardized)")
    plt.ylabel("Technical Score (standardized)")
    plt.grid(True)
    
    if filename:
        plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot the decision boundary for initial model
plot_decision_boundary(model, X_train_scaled, y_train, 
                      title="Job Application Evaluation: Linear Kernel Decision Boundary",
                      filename="linear_kernel_decision_boundary")

# Try different kernels
kernels = ['rbf', 'poly', 'sigmoid']
for kernel in kernels:
    print(f"\nTrying kernel: {kernel}")
    model = SVC(kernel=kernel)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {kernel} kernel: {accuracy:.2f}")

# Enhanced Parameter tuning with visualization
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)
print(f"Best accuracy: {grid_search.best_score_:.2f}")

# Visualize parameter performance
results = grid_search.cv_results_
C_values = param_grid['C']
gamma_values = param_grid['gamma']
kernels = param_grid['kernel']

# Create a 3D plot for each kernel
for kernel in kernels:
    plt.figure(figsize=(10, 6))
    scores = []
    for C in C_values:
        for gamma in gamma_values:
            mask = (results['param_C'] == C) & (results['param_gamma'] == gamma) & (results['param_kernel'] == kernel)
            if any(mask):
                scores.append(results['mean_test_score'][mask][0])
    
    if scores:
        scores = np.array(scores).reshape(len(C_values), len(gamma_values))
        plt.imshow(scores, cmap='viridis')
        plt.colorbar(label='Accuracy')
        plt.xticks(np.arange(len(gamma_values)), gamma_values)
        plt.yticks(np.arange(len(C_values)), C_values)
        plt.xlabel('Gamma')
        plt.ylabel('C')
        plt.title(f'Kernel: {kernel} - Parameter Performance')
        plt.savefig(f'plots/{kernel}_parameter_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

# Train final model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Evaluate best model
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"\nBest model accuracy (test set): {accuracy_best:.2f}")
print("\nBest model classification report:")
print(classification_report(y_test, y_pred_best))

# Plot decision boundary for best model
plot_decision_boundary(best_model, X_train_scaled, y_train,
                      title="Job Application Evaluation: Best Model Decision Boundary",
                      filename="best_model_decision_boundary")

# Function to make predictions
def predict_application(experience_years, technical_score):
    # Scale the input
    input_data = np.array([[experience_years, technical_score]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Check if the prediction matches our criteria
    if experience_years < 2 and technical_score < 60:
        return "Not hired"
    else:
        return "Hired"

# Example prediction
print("\nExample prediction:")
experience = float(input("Enter experience years (0-10): "))
score = float(input("Enter technical score (0-100): "))
result = predict_application(experience, score)
print(f"Result: {result}") 