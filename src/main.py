from models.svm_model import JobApplicantSVM

def main():
    # Initialize the model
    model = JobApplicantSVM()
    
    # Generate and prepare data
    X, y = model.generate_data()
    model.prepare_data(X, y)
    
    # Train initial model
    model.train_model()
    
    # Evaluate initial model
    print("\nInitial Model Evaluation:")
    model.evaluate_model()
    
    # Plot initial decision boundary
    model.plot_decision_boundary(
        title="İşe Alım Değerlendirmesi: Linear Kernel Karar Sınırı",
        filename="linear_kernel_decision_boundary"
    )
    
    # Tune parameters
    print("\nParameter Tuning:")
    best_params = model.tune_parameters()
    
    # Plot best model decision boundary
    model.plot_decision_boundary(
        title="İşe Alım Değerlendirmesi: En İyi Model Karar Sınırı",
        filename="best_model_decision_boundary"
    )
    
    # Example prediction
    print("\nÖrnek tahmin:")
    tecrube = float(input("Tecrübe yılı girin (0-10): "))
    puan = float(input("Teknik puan girin (0-100): "))
    result = model.predict(tecrube, puan)
    print(f"Sonuç: {result}")

if __name__ == "__main__":
    main() 