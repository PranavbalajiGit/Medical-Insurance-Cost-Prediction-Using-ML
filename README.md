# Medical Insurance Cost Prediction

A machine learning project that predicts medical insurance costs based on various personal and demographic factors using **Linear Regression**.

---

## 📑 Table of Contents
- Overview  
- Dataset  
- Features  
- Technologies Used  
- Data Analysis  
- Model Implementation  
- Results  
- Installation  
- Usage  
- Project Structure  
- Future Improvements  
- Contributing  
- License  
- Acknowledgments  

---

## 📌 Overview
This project implements a **Linear Regression** model to predict medical insurance costs.  
The model analyzes various factors such as **age, BMI, number of children, smoking status, gender, and region** to estimate insurance charges for individuals.

---

## 📊 Dataset
The dataset (`insurance.csv`) contains **1,338 records** with the following attributes:

| Feature   | Description                                   | Type         |
|----------|-----------------------------------------------|--------------|
| age      | Age of the insured person                     | Numerical    |
| sex      | Gender (male/female)                          | Categorical  |
| bmi      | Body Mass Index                               | Numerical    |
| children | Number of children/dependents                 | Numerical    |
| smoker   | Smoking status (yes/no)                       | Categorical  |
| region   | Residential region                            | Categorical  |
| charges  | Medical insurance cost (Target Variable)      | Numerical    |

### Dataset Statistics
- **Shape:** 1,338 rows × 7 columns  
- **Missing Values:** None  
- **Gender Distribution:** 676 males, 662 females  
- **Smoker Distribution:** 1,064 non-smokers, 274 smokers  
- **Region Distribution:** Relatively balanced across all four regions  

---

## 🧩 Features

### Data Preprocessing

**Categorical Encoding**
- Gender: `male = 0`, `female = 1`
- Smoker: `yes = 0`, `no = 1`
- Region:
  - southeast = 0  
  - southwest = 1  
  - northeast = 2  
  - northwest = 3  

**Feature Engineering**
- All categorical variables converted to numerical format for model compatibility

---

## 🛠 Technologies Used
- **Python 3.x**
- **NumPy** – Numerical computing  
- **Pandas** – Data manipulation and analysis  
- **Matplotlib** – Data visualization  
- **Seaborn** – Statistical data visualization  
- **Scikit-learn**
  - `LinearRegression`
  - `train_test_split`
  - `r2_score`

---

## 🔍 Data Analysis

### Exploratory Data Analysis (EDA)

- **Age Distribution**
  - Visualized using distribution plots
  - Shows the spread of ages across the dataset

- **Gender Distribution**
  - Nearly equal distribution between males and females
  - Visualized using count plots

- **BMI Distribution**
  - Normal distribution pattern
  - Centered around BMI ≈ 30

- **Children Distribution**
  - Most insured individuals have 0–2 children
  - Maximum of 5 children

- **Smoker Status**
  - Non-smokers: 79.5%
  - Smokers: 20.5%

- **Regional Distribution**
  - Fairly balanced across regions
  - Southeast slightly higher representation

- **Charges Distribution**
  - Right-skewed
  - Majority of values in lower range with few high outliers

---

## ⚙️ Model Implementation

### Data Splitting
- **Training Set:** 80% (1,070 samples)
- **Test Set:** 20% (268 samples)
- **Random State:** 2 (for reproducibility)

### Algorithm
**Linear Regression** was chosen because:
- Simple and interpretable
- Suitable for continuous target variables
- Captures linear relationships
- Fast training and prediction

### Model Training
```python
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

## 📈 Results

### Model Performance

| Metric  | Training Data | Test Data |
|--------|---------------|-----------|
| R² Score | 0.7515 | 0.7447 |

### Interpretation
- **Training R²:** Explains ~75.15% variance  
- **Test R²:** Explains ~74.47% variance  
- **Minimal overfitting:** Difference < 1%  
- **Good generalization capability**

### Key Insights
- Consistent performance across training and test datasets  
- Strong predictors include **smoking status, age, and BMI**  
- Model is reliable, interpretable, and suitable for baseline prediction  

---

## 🧪 Installation

### Clone the Repository
```bash
git clone https://github.com/PranavbalajiGit/Medical-Insurance-Cost-Prediction-Using-ML
cd Medical-Insurance-Cost-Prediction-Using-ML
```

**Install Required Packages**
pip install numpy pandas matplotlib seaborn scikit-learn

**Dataset**
Ensure the dataset file is present:
insurance.csv


Run all cells to:

Load and explore data
Visualize distributions
Preprocess features
Train the model
Evaluate performance

**Making Predictions**

# Example prediction for a new patient
new_patient = [[35, 0, 28.5, 2, 1, 1]]  # age, sex, bmi, children, smoker, region
predicted_cost = regressor.predict(new_patient)
print(f"Predicted Insurance Cost: ${predicted_cost[0]:.2f}")


**📂 Project Structure**
medical-insurance-prediction/
│
├── medical_insurance_cost.ipynb    # Main Jupyter notebook
├── insurance.csv                   # Dataset
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies

**🤝 Contributing**

Contributions are welcome!
Feel free to submit a Pull Request.

.

**📜 License**
This project is open source and available under the MIT License.

**Note:**
This project is for educational purposes only.
Always consult insurance professionals for real-world insurance decisions.