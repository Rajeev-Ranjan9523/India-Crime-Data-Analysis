# 🇮🇳 India Crime Data Analysis Using Python and Machine Learning

## 📌 Project Overview

India Crime Data Analysis is a Python-based project that analyses crime records across different states and years in India.

The main aim of this project is to understand crime patterns, identify crime trends, compare different states, and apply machine learning techniques for crime prediction and analysis.

## 🎯 Problem Statement

A large amount of crime data is available across different states and years, but understanding crime patterns manually is difficult.

This project uses Python, data visualization, and machine learning techniques to analyse crime data, identify important patterns, group similar records, and predict crime-related outcomes.

## 📂 Dataset Information

- Dataset: India Crime Dataset
- Total Rows: 15,264
- Total Columns: 8
- States/UTs: 36
- Years: 2018–2020

*Note: These figures describe the original dataset before data cleaning.*

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 📊 Project Implementation

### 1. Data Cleaning
- Removed duplicate records.
- Handled missing values.
- Converted columns into suitable data types.
- Corrected negative values in crime-related columns.

### 2. Exploratory Data Analysis (EDA)
- Analysed crime records by state and year.
- Identified states with the highest recorded incidents.
- Analysed different crime categories.
- Created graphs to understand crime trends.

### 3. Regression Analysis
Applied different regression algorithms to predict crime incidents:

- Linear Regression
- Polynomial Regression
- Multiple Linear Regression
- Random Forest Regressor
- Bagging Regressor
- Gradient Boosting Regressor
- MLP Regressor

### 4. Classification
Used classification algorithms to classify crime records into low and high incident groups:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- MLP Classifier

### 5. Clustering Analysis
- Applied K-Means Clustering to group similar state-year crime records.
- Used the Elbow Method to examine a suitable number of clusters.
- Applied Agglomerative Clustering with different linkage methods.

### 6. Association Pattern Analysis
Identified crime subcategory co-occurrence patterns within state-year groups using support and confidence.

### 7. Principal Component Analysis (PCA)
Applied PCA to reduce the features into two principal components and visualize the data in a 2D scatter plot.

### 8. Model Performance Evaluation
Compared machine learning models using different evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- MAE
- MSE
- RMSE
- R² Score
- ROC-AUC

## 📈 Project Outputs

The project generates:
- Crime analysis graphs
- State-wise and year-wise crime summaries
- Regression model results
- Classification performance reports
- Confusion matrix
- Clustering visualizations
- Elbow Method graph
- PCA scatter plot
- Model comparison results

## ▶️ How to Run the Project

**Step 1:** Clone the repository.

```bash
git clone https://github.com/YOUR-USERNAME/India-Crime-Data-Analysis.git
```

**Step 2:** Open the project folder.

```bash
cd India-Crime-Data-Analysis
```

**Step 3:** Install the required libraries.

```bash
pip install pandas numpy matplotlib scikit-learn
```

**Step 4:** Update the dataset path in the Python file according to your computer.

**Step 5:** Run the Python file.

```bash
python CA2Project_Int234.py
```

## 💡 Key Learning Outcomes

- Learned how to clean and analyse real-world datasets.
- Understood data visualization using Matplotlib.
- Implemented regression and classification algorithms.
- Learned K-Means and hierarchical clustering.
- Understood the Elbow Method and PCA.
- Compared machine learning models using evaluation metrics.

## ⚠️ Limitations

The analysis is based on recorded crime data. Actual crime incidents may differ from officially recorded figures. The predictions are based on historical data and should not be treated as confirmed future crime statistics.

## 👨‍💻 Author

**Rajeev Ranjan**

B.Tech – Computer Science and Engineering (Data Science)

GitHub: https://github.com/Rajeev-Ranjan9523

---

⭐ If you find this project useful, feel free to star the repository!
