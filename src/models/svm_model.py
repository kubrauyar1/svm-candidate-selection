import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
import os

class JobApplicantSVM:
    def __init__(self, n_samples=200, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Set random seeds
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)

    def generate_data(self):
        """Generate synthetic applicant data"""
        tecrube_yili = np.random.uniform(0, 10, self.n_samples)
        teknik_puan = np.random.uniform(0, 100, self.n_samples)

        # Create labels based on criteria
        labels = []
        for tecrube, puan in zip(tecrube_yili, teknik_puan):
            if tecrube < 2 and puan < 60:
                labels.append(1)  # Not hired
            else:
                labels.append(0)  # Hired

        # Prepare the data
        X = np.column_stack((tecrube_yili, teknik_puan))
        y = np.array(labels)
        
        return X, y

    def prepare_data(self, X, y):
        """Split and scale the data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, kernel='linear'):
        """Train the SVM model"""
        self.model = SVC(kernel=kernel)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate_model(self):
        """Evaluate the model performance"""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        return accuracy

    def plot_decision_boundary(self, title="İşe Alım Değerlendirmesi: SVM Karar Sınırı", filename=None):
        """Plot the decision boundary"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, 
                   cmap='bwr', s=60, edgecolors='k', alpha=0.7)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.model.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
                   linestyles=['--', '-', '--'])

        ax.scatter(self.model.support_vectors_[:, 0], self.model.support_vectors_[:, 1],
                   s=150, linewidth=1.5, facecolors='none', edgecolors='k')

        plt.title(title)
        plt.xlabel("Tecrübe Yılı (standardize)")
        plt.ylabel("Teknik Puan (standardize)")
        plt.grid(True)
        
        if filename:
            plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def tune_parameters(self):
        """Perform hyperparameter tuning"""
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }

        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        print("\nEn iyi parametreler:")
        print(grid_search.best_params_)
        print(f"En iyi doğruluk: {grid_search.best_score_:.2f}")

        # Visualize parameter performance
        self._plot_parameter_performance(grid_search, param_grid)

        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def _plot_parameter_performance(self, grid_search, param_grid):
        """Plot parameter performance for each kernel"""
        results = grid_search.cv_results_
        C_values = param_grid['C']
        gamma_values = param_grid['gamma']
        kernels = param_grid['kernel']

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
                plt.colorbar(label='Doğruluk')
                plt.xticks(np.arange(len(gamma_values)), gamma_values)
                plt.yticks(np.arange(len(C_values)), C_values)
                plt.xlabel('Gamma')
                plt.ylabel('C')
                plt.title(f'Kernel: {kernel} - Parametre Performansı')
                plt.savefig(f'plots/{kernel}_parameter_performance.png', dpi=300, bbox_inches='tight')
                plt.show()

    def predict(self, tecrube_yili, teknik_puan):
        """Make a prediction for a new applicant"""
        input_data = np.array([[tecrube_yili, teknik_puan]])
        input_scaled = self.scaler.transform(input_data)
        
        if tecrube_yili < 2 and teknik_puan < 60:
            return "İşe alınmadı"
        else:
            return "İşe alındı" 