# Job Applicant Selection: Application Evaluation with SVM

This project contains a machine learning model that predicts whether job applicants should be hired based on their years of experience and technical exam score for software developer positions.

## Project Structure

```
.
├── src/                    # Source code
│   ├── __init__.py        # Package definition
│   ├── main.py            # Main execution file
│   ├── models/            # Model classes
│   │   └── svm_model.py   # SVM model
│   ├── api/               # API code
│   │   └── app.py         # FastAPI application
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── docs/                  # Documentation
├── tests/                 # Test files
├── plots/                 # Visualizations
├── requirements.txt       # Dependencies
└── README.md             # Project documentation
```

## Features

- Job applicant evaluation using SVM (Support Vector Machine)
- Experience with different kernel functions (linear, rbf, poly, sigmoid)
- Hyperparameter tuning (C and gamma parameters)
- REST API service with FastAPI
- Visualization and decision boundary analysis
- Automatic image saving and reporting

## Installation

1. Clone the project:
```bash
git clone https://github.com/username/project_name.git
cd project_name
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Model Training and Evaluation

To run the main model:
```bash
python src/main.py
```

This command will:
- Generate synthetic data
- Train the model
- Try different kernel functions
- Perform hyperparameter tuning
- Visualize decision boundaries
- Take user input for predictions
- Save all visualizations to the `plots/` directory

### API Service

To start the API service:
```bash
uvicorn src.api.app:app --reload
```

To test the API:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"experience_years": 3, "technical_score": 75}'
```

## Model Criteria

Applicants are evaluated based on the following criteria:
- Experience < 2 years AND technical score < 60: Not hired (1)
- Other cases: Hired (0)

## Visualizations

Visualizations created during model training are stored in the `plots/` directory:
- `linear_kernel_decision_boundary.png`: Decision boundary for linear kernel
- `rbf_parameter_performance.png`: Parameter performance for RBF kernel
- `poly_parameter_performance.png`: Parameter performance for polynomial kernel
- `sigmoid_parameter_performance.png`: Parameter performance for sigmoid kernel
- `best_model_decision_boundary.png`: Decision boundary for best model

## API Endpoints

- `GET /`: Information about the API
- `POST /predict`: Evaluates job applicants

## Requirements

- Python 3.8+
- numpy
- matplotlib
- scikit-learn
- fastapi
- uvicorn

## Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature/newFeature`)
3. Commit your changes (`git commit -am 'Added new feature'`)
4. Push to the branch (`git push origin feature/newFeature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 