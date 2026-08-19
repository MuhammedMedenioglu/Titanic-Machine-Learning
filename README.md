# 🚢 Titanic Survival Prediction — End-to-End Machine Learning Project

A complete machine learning pipeline built on the classic Titanic dataset. This project covers exploratory data analysis, feature engineering, training of 6 individual classifiers, and a final soft-voting ensemble — with full evaluation including confusion matrices and probability distribution plots for every model.

\---

## 📋 Table of Contents

* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [Feature Engineering](#-feature-engineering)
* [Models \& Results](#-models--results)
* [Final Ensemble](#-final-ensemble)
* [How to Run](#-how-to-run)
* [Dependencies](#-dependencies)

\---

## 📦 Dataset

**Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

|Property|Value|
|-|-|
|Rows|891 passengers|
|Target column|`Survived` (0 = No, 1 = Yes)|
|Key features|`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`|

\---

## 🗂 Project Structure

```
titanic-ml/
│
├── Titanic-Dataset.csv
├── titanic.py                  # Main pipeline script
│
├── plots/
│   ├── Survival\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Count\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_by\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Gender.png
│   ├── Survival\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Count\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_by\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Passenger\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Class.png
│   │
│   ├── logistic\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_regression\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_confusion\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_matrix.png
│   ├── logistic\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_regression\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prediction\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_probability\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_distribution.png
│   ├── KNN\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_confusion\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_matrix.png
│   ├── KNN\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prediction\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_probability\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_distribution.png
│   ├── Naive\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Bayes\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_confusion\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_matrix.png
│   ├── Naive\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Bayes\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prediction\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_probability\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_distribution.png
│   ├── SVM\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_confusion\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_matrix.png
│   ├── SVM\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prediction\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_probability\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_distribution.png
│   ├── Random\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Forest\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_confusion\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_matrix.png
│   ├── Random\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Forest\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prediction\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_probability\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_distribution.png
│   ├── xgboost\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_confusion\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_matrix.png
│   ├── xgboost\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prediction\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_probability\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_distribution.png
│   ├── ensemble\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_confusion\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_matrix.png
│   └── ensemble\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prediction\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_probability\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_distribution.png
│
└── README.md
```

\---

## 📊 Exploratory Data Analysis

Two key survival patterns were identified in the EDA phase:

**Survival by Gender** — Women survived at a dramatically higher rate (\~74%) compared to men (\~19%), reflecting the "women and children first" evacuation policy.

**Survival by Passenger Class** — First-class passengers had a significantly higher survival rate than third-class passengers, highlighting the role of socioeconomic status in the disaster.

||Survived|Died|
|-|-|-|
|Female|233|81|
|Male|109|468|

||Survived|Died|
|-|-|-|
|1st Class|136|80|
|2nd Class|87|97|
|3rd Class|119|372|

\---

## 🛠 Feature Engineering

### Handling Missing Values

* **Age:** Filled with the **median** age to minimize the effect of outliers.
* **Embarked:** Filled with the **mode** (most frequent port).
* **Cabin:** Dropped due to a high proportion of missing values (\~77%).

### Categorical Encoding

|Column|Method|Detail|
|-|-|-|
|`Sex`|Binary Encoding|`female → 1`, `male → 0`|
|`Embarked`|One-Hot Encoding|`get\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_dummies(drop\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_first=True)` → `Embarked\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Q`, `Embarked\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_S`|

> \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*Note on the Dummy Variable Trap:\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\* `drop\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_first=True` drops `Embarked\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_C`. A value of `0` in both `Embarked\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Q` and `Embarked\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_S` implicitly means the passenger embarked at Cherbourg ('C'), preventing multicollinearity.

### Dropped Columns

`Name`, `Ticket`, `Cabin`, `PassengerId` — non-informative for the model.

\---

## 🤖 Models \& Results

All models were evaluated on the same held-out test set. KNN and SVM use StandardScaler; tree-based models do not require scaling.

|Model|Accuracy|Precision|Recall|F1 Score|
|-|-|-|-|-|
|Logistic Regression|80.4%|79.3%|66.7%|72.4%|
|KNN (Scaled)|81.6%|80.0%|69.6%|74.4%|
|Naive Bayes|78.2%|72.7%|69.6%|71.1%|
|SVM (Scaled)|81.0%|**84.3%**|62.3%|71.7%|
|Random Forest (Optimized)|79.9%|77.0%|68.1%|72.3%|
|XGBoost|79.3%|79.6%|62.3%|69.9%|
|**Final Ensemble**|**82.7%**|82.8%|**69.6%**|**75.6%**|

Each model's confusion matrix and probability distribution plot is available in the `plots/` folder.

\---

## 🏆 Final Ensemble

The final model is a **soft-voting ensemble** combining all 6 classifiers. By averaging predicted probabilities across models, it reduces the variance of any single algorithm and achieves the best overall performance on all four metrics.

**Confusion Matrix (Ensemble):**

||Predicted Dead|Predicted Survived|
|-|-|-|
|**Actual Dead**|100 ✅|10 ❌|
|**Actual Survived**|21 ❌|48 ✅|

* **True Negative Rate (Specificity):** 90.9%
* **True Positive Rate (Recall):** 69.6%

\---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/titanic-ml.git
cd titanic-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python titanic.py
```

All plots will be saved automatically to the `plots/` folder.

\---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

\---

## 📝 Notes

* Train/test split: **80/20**, stratified by `Survived`
* Random state fixed at `42` for reproducibility
* Hyperparameter tuning on Random Forest via `GridSearchCV`

\---

*Built with Python 3 · scikit-learn · XGBoost*

