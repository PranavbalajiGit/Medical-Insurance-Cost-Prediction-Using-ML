Medical Insurance Cost Prediction
A machine learning project that predicts medical insurance costs based on various personal and demographic factors using Linear Regression.
Table of Contents

Overview
Dataset
Features
Technologies Used
Data Analysis
Model Implementation
Results
Installation
Usage

Overview
This project implements a Linear Regression model to predict medical insurance costs. The model analyzes various factors such as age, BMI, number of children, smoking status, gender, and region to estimate insurance charges for individuals.
Dataset
The dataset (insurance.csv) contains 1,338 records with the following attributes:
FeatureDescriptionTypeageAge of the insured personNumericalsexGender (male/female)CategoricalbmiBody Mass IndexNumericalchildrenNumber of children/dependentsNumericalsmokerSmoking status (yes/no)CategoricalregionResidential region (northeast, northwest, southeast, southwest)CategoricalchargesMedical insurance cost (Target Variable)Numerical
Dataset Statistics

Shape: 1,338 rows × 7 columns
No missing values in any column
Gender Distribution: 676 males, 662 females
Smoker Distribution: 1,064 non-smokers, 274 smokers
Region Distribution: Relatively balanced across all four regions

Features
Data Preprocessing

Categorical Encoding:

Gender: male = 0, female = 1
Smoker: yes = 0, no = 1
Region: southeast = 0, southwest = 1, northeast = 2, northwest = 3


Feature Engineering: All categorical variables converted to numerical format for model compatibility

Technologies Used
python- Python 3.x
- NumPy - Numerical computing
- Pandas - Data manipulation and analysis
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- Scikit-learn - Machine learning library
  - LinearRegression - Model
  - train_test_split - Data splitting
  - r2_score - Model evaluation
Data Analysis
Exploratory Data Analysis (EDA)

Age Distribution:

Visualized using distribution plot
Shows the spread of ages across the dataset


Gender Distribution:

Nearly equal distribution between males and females
Visualized using count plot


BMI Distribution:

Normal distribution pattern observed
Centered around 30 (average BMI)


Children Distribution:

Most insured individuals have 0-2 children
Maximum of 5 children in the dataset


Smoker Status:

Majority are non-smokers (79.5%)
Smokers represent 20.5% of the dataset


Regional Distribution:

Fairly balanced across all four regions
Southeast has slightly more representation (364 records)


Charges Distribution:

Right-skewed distribution
Most charges fall in the lower range with some high outliers



Model Implementation
Data Splitting

Training Set: 80% (1,070 samples)
Test Set: 20% (268 samples)
Random State: 2 (for reproducibility)

Algorithm
Linear Regression was chosen for this regression task because:

Simple and interpretable
Works well with continuous target variables
Establishes linear relationships between features and target
Fast training and prediction

Model Training
pythonregressor = LinearRegression()
regressor.fit(X_train, Y_train)
Results
Model Performance
MetricTraining DataTest DataR² Score0.75150.7447
Interpretation

Training R² Score: 75.15%

The model explains approximately 75% of the variance in insurance costs on training data


Test R² Score: 74.47%

The model maintains consistent performance on unseen data
Minimal overfitting (difference < 1%)
Good generalization capability



Key Insights

Model Reliability: The close R² scores between training and test sets indicate the model generalizes well to new data
Predictive Power: With an R² score of ~74%, the model can reasonably predict insurance costs based on the given features
Feature Importance: The linear relationship suggests that features like smoking status, age, and BMI are likely strong predictors of insurance costs

Installation

Clone the repository:

bashgit clone https://github.com/yourusername/medical-insurance-prediction.git
cd medical-insurance-prediction

Install required packages:

bashpip install numpy pandas matplotlib seaborn scikit-learn
```

3. Ensure you have the dataset:
```
insurance.csv
Usage
Running the Notebook

Open the Jupyter notebook:

bashjupyter notebook medical_insurance_cost.ipynb

Run all cells sequentially to:

Load and explore the data
Visualize distributions
Preprocess features
Train the model
Evaluate performance



Making Predictions
python# Example prediction for a new patient
new_patient = [[35, 0, 28.5, 2, 1, 1]]  # age, sex, bmi, children, smoker, region
predicted_cost = regressor.predict(new_patient)
print(f"Predicted Insurance Cost: ${predicted_cost[0]:.2f}")
```

## Project Structure
```
medical-insurance-prediction/
│
├── medical_insurance_cost.ipynb    # Main Jupyter notebook
├── insurance.csv                   # Dataset
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
Future Improvements

Feature Engineering:

Create interaction features (e.g., smoker × age, smoker × BMI)
Polynomial features for non-linear relationships


Model Enhancement:

Try advanced algorithms (Random Forest, Gradient Boosting)
Hyperparameter tuning
Cross-validation for robust evaluation


Additional Analysis:

Feature importance analysis
Residual analysis
Outlier detection and treatment


Deployment:

Create a web interface for predictions
Deploy model as REST API
Build interactive dashboard



Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is open source and available under the MIT License.
Acknowledgments

Dataset source: [Insert dataset source if applicable]
Inspiration from various insurance cost prediction projects
Thanks to the scikit-learn and pandas communities


Note: This is an educational project for demonstrating machine learning techniques in predicting insurance costs. Always consult with insurance professionals for actual insurance decisions.