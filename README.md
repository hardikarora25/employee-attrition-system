# 💼 Employee Attrition Prediction System

An end-to-end Machine Learning project to predict employee turnover, allowing HR departments to proactively identify at-risk employees and take retention measures.

## 📌 Business Problem
Employee attrition is a major cost for modern organizations. Finding, interviewing, and training new hires requires significant time and capital. The goal of this project is to use historical HR data (demographics, job satisfaction, salary, overtime, etc.) to build a predictive model that identifies whether an employee is at a **High Risk** of leaving the company. 

Because attrition datasets are highly imbalanced (most employees stay), this project heavily focuses on handling class imbalance and optimizing for the **Recall** metric to ensure we don't miss employees who are actually leaving.

---

## 🚀 Key Features & Achievements
- **End-to-End Pipeline:** Covers raw data cleansing, Exploratory Data Analysis (EDA), feature encoding, model training, hyperparameter tuning, and a web deployment.
- **Handled Imbalanced Data:** Utilized `class_weight='balanced'` and Grid Search hyperparameter tuning to force the model to learn the minority class.
- **Massive Recall Improvement:** Successfully boosted the model's ability to catch at-risk employees from a baseline of ~8.5% to **~60%**.
- **Interactive Web Dashboard:** Built a Streamlit application featuring both a Single Employee manual predictor and a Bulk CSV processing tool.

---

## 📊 Model Evaluation & Selection
Three baseline models were trained: Logistic Regression, Decision Tree, and Random Forest. After baseline testing, the models were fine-tuned using `GridSearchCV` specifically optimizing for the **Recall** score.

**Final Test Set Performance (Tuned Random Forest):**
- **Accuracy:** 79.25%
- **Precision:** 40.00%
- **Recall:** 59.57%
- **F1-Score:** 47.86%

*Note: In HR analytics, Recall is prioritized over Precision because the cost of a False Negative (missing an employee who leaves) is much higher than a False Positive (having an extra meeting with an employee who stays).*

---

## 📂 Project Structure
```text
employee_attrition_system/
│
├── data/                             # Datasets (Raw, Cleaned, and ML-Ready)
│   ├── attrition_clean.csv           # Baseline cleaned data (missing values handled)
│   └── attrition_ml_ready.csv        # Fully encoded and scaled data for training
│
├── models/                           # Serialized models and scalers
│   ├── best_logistic_regression.pkl  # Baseline model
│   ├── final_optimized_rf_model.pkl  # Final tuned Random Forest model
│   └── scaler.pkl                    # StandardScaler for consistent production inputs
│
├── notebooks/                        # Jupyter Notebooks (The Data Science Workflow)
│   ├── day1_attrition_exploration.ipynb
│   ├── attrition_eda_and_preprocessing.ipynb
│   ├── attrition_model_development.ipynb
│   └── attrition_model_optimization.ipynb
│
├── src/                              # Source code for production
│   └── app.py                        # Streamlit Web Application
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🖥️ Streamlit Web Deployment

To make the model accessible to non-technical HR staff, a web dashboard was built using **Streamlit**. It features two modes:
1. **Single Employee Predictor:** Allows managers to manually adjust sliders (e.g., Salary, Age, Overtime) and see real-time updates to the Attrition Probability gauge. Includes a dynamic **Feature Importance** chart.
2. **Bulk CSV Predictor:** Allows HR teams to upload a spreadsheet of thousands of employees, process them all simultaneously, and download a report with predicted attrition flags.

### How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hardikarora25/employee-attrition-system.git
   cd employee-attrition-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web app:**
   ```bash
   streamlit run src/app.py
   ```
   *The application will automatically open in your browser at `http://localhost:8501`.*