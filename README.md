# Benchmarking-and-Optimization-of-Machine-Learning

# Benchmarking and Optimization of Machine Learning Classification Models

A comparative machine learning project that evaluates and optimizes multiple supervised classification algorithms on the **Iris** and **Wine** datasets. The project focuses on model benchmarking, feature preprocessing, stratified cross-validation, hyperparameter optimization, and comprehensive performance evaluation.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Datasets](#datasets)
* [Technologies Used](#technologies-used)
* [Machine Learning Models](#machine-learning-models)
* [Project Workflow](#project-workflow)
* [Data Preprocessing](#data-preprocessing)
* [Cross-Validation](#cross-validation)
* [Hyperparameter Optimization](#hyperparameter-optimization)
* [Evaluation Metrics](#evaluation-metrics)
* [Decision Tree Analysis](#decision-tree-analysis)
* [Results](#results)
* [Key Findings](#key-findings)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Future Improvements](#future-improvements)
* [Learning Outcomes](#learning-outcomes)
* [Conclusion](#conclusion)
* [Author](#author)

---

## Project Overview

Machine learning classification problems can often be solved using multiple algorithms, but different algorithms may perform differently depending on the characteristics of the dataset.

This project performs a systematic comparison of several supervised machine learning algorithms using two well-known multiclass classification datasets:

* **Iris Dataset**
* **Wine Dataset**

The project implements multiple classification algorithms, performs hyperparameter optimization using **GridSearchCV**, and evaluates the optimized models using an independent test set.

The analysis includes:

* Exploratory Data Analysis
* Feature distributions
* Correlation analysis
* Train/test splitting
* Feature scaling
* Stratified 5-fold cross-validation
* Hyperparameter optimization
* Model comparison
* Confusion matrices
* Classification reports
* Decision Tree analysis
* Gini vs. Entropy comparison
* Random Forest feature importance

---

## Objectives

The primary objectives of this project are:

1. Implement multiple supervised machine learning classification algorithms.
2. Compare their performance on the Iris and Wine datasets.
3. Apply appropriate preprocessing and feature scaling.
4. Use stratified cross-validation for reliable model selection.
5. Optimize model hyperparameters using `GridSearchCV`.
6. Evaluate optimized models using multiple performance metrics.
7. Analyze classification errors using confusion matrices.
8. Compare Decision Trees using Gini and Entropy splitting criteria.
9. Identify important features using Random Forest.
10. Determine the best-performing models for each dataset.

---

# Datasets

## 1. Iris Dataset

The Iris dataset is a classic multiclass classification dataset containing measurements of iris flowers from three different species.

### Dataset Characteristics

| Property     |                     Value |
| ------------ | ------------------------: |
| Samples      |                       150 |
| Features     |                         4 |
| Classes      |                         3 |
| Feature Type |                 Numerical |
| Problem Type | Multiclass Classification |

### Features

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Classes

* Setosa
* Versicolor
* Virginica

---

## 2. Wine Dataset

The Wine dataset contains chemical analysis results of wines belonging to three different cultivars.

### Dataset Characteristics

| Property     |                     Value |
| ------------ | ------------------------: |
| Samples      |                       178 |
| Features     |                        13 |
| Classes      |                         3 |
| Feature Type |                 Numerical |
| Problem Type | Multiclass Classification |

### Example Features

* Alcohol
* Malic Acid
* Ash
* Alcalinity of Ash
* Magnesium
* Total Phenols
* Flavanoids
* Nonflavanoid Phenols
* Proanthocyanins
* Color Intensity
* Hue
* OD280/OD315 of Diluted Wines
* Proline

---

# Technologies Used

The project is implemented using Python and the following libraries:

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| NumPy        | Numerical computation          |
| Pandas       | Data manipulation and analysis |
| Scikit-learn | Machine learning               |
| Matplotlib   | Data visualization             |
| Seaborn      | Statistical visualization      |
| Kaggle       | Notebook execution environment |

---

# Machine Learning Models

Seven classification algorithms were implemented and compared.

## 1. Logistic Regression

A linear classification algorithm that models the probability of class membership.

---

## 2. K-Nearest Neighbors

A distance-based classification algorithm that predicts the class of a sample based on its nearest neighbors.

---

## 3. Gaussian Naive Bayes

A probabilistic classifier based on Bayes' theorem that assumes continuous features follow a Gaussian distribution.

---

## 4. Support Vector Machine

A classification algorithm that attempts to find an optimal decision boundary between classes.

The project uses both linear and RBF kernels during hyperparameter optimization.

---

## 5. Decision Tree

A tree-based classification algorithm that recursively divides the dataset using feature-based decision rules.

Both **Gini** and **Entropy** criteria are investigated.

---

## 6. Random Forest

An ensemble learning method that combines predictions from multiple Decision Trees to improve generalization.

---

## 7. Gradient Boosting

An ensemble learning algorithm that builds models sequentially, where each new model attempts to correct errors made by previous models.

---

# Project Workflow

```text
                       ┌──────────────────┐
                       │  Iris & Wine     │
                       │     Datasets     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Exploratory Data │
                       │     Analysis     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Data Cleaning   │
                       │ & Preprocessing  │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Train/Test Split │
                       │     80 / 20      │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Feature Scaling  │
                       │  Where Required  │
                       └────────┬─────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │       ML Model Training       │
                ├───────────────────────────────┤
                │ Logistic Regression           │
                │ KNN                           │
                │ Gaussian Naive Bayes          │
                │ SVM                           │
                │ Decision Tree                 │
                │ Random Forest                 │
                │ Gradient Boosting             │
                └───────────────┬───────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Stratified 5-Fold│
                       │ Cross Validation │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   GridSearchCV   │
                       │ Hyperparameter   │
                       │   Optimization   │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Best Model       │
                       │ Selection        │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Independent Test │
                       │    Evaluation    │
                       └────────┬─────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
          Accuracy          Precision         Recall
               │                │                │
               └────────────────┼────────────────┘
                                ▼
                           F1 Score
                                │
                                ▼
                       Confusion Matrices
                                │
                                ▼
                       Model Comparison
```

---

# Data Preprocessing

An **80/20 train-test split** is used for each dataset.

The split is stratified to preserve the class distribution between the training and testing sets.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

### Feature Scaling

`StandardScaler` is used for algorithms that are sensitive to feature magnitude:

* Logistic Regression
* KNN
* SVM

Scaling is implemented inside a Scikit-learn `Pipeline`.

This prevents information from the test set from leaking into the training process during cross-validation.

---

# Cross-Validation

The project uses **Stratified 5-Fold Cross-Validation**.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Each training dataset is divided into five folds.

```text
             Training Dataset
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    Fold 1        Fold 2       Fold 3
       │            │            │
       └────────────┼────────────┘
                    │
               Fold 4 / Fold 5
                    │
                    ▼
             Average CV Score
```

Stratification helps maintain a similar proportion of classes in each fold.

---

# Hyperparameter Optimization

`GridSearchCV` is used to search through predefined hyperparameter combinations.

The model configuration producing the highest cross-validation accuracy is selected.

### Logistic Regression

Parameters explored:

* `C`
* `solver`

### KNN

Parameters explored:

* `n_neighbors`
* `weights`
* `metric`

### Gaussian Naive Bayes

Parameters explored:

* `var_smoothing`

### SVM

Parameters explored:

* `C`
* `gamma`
* `kernel`

### Decision Tree

Parameters explored:

* `criterion`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

### Random Forest

Parameters explored:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

### Gradient Boosting

Parameters explored:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `subsample`

---

# Evaluation Metrics

The optimized models are evaluated using several metrics.

## Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Precision

Measures how many samples predicted as a particular class actually belong to that class.

---

## Recall

Measures how many samples belonging to a particular class were correctly identified.

---

## F1-Score

The F1-score combines Precision and Recall into a single metric.

---

## Confusion Matrix

Confusion matrices are generated for every model and dataset.

They provide a detailed view of the model's predictions across all classes.

---

# Decision Tree Analysis

Decision Trees are analyzed using two splitting criteria:

## Gini

Gini impurity measures the degree of class impurity within a node.

A lower impurity indicates a more homogeneous node.

---

## Entropy

Entropy measures the uncertainty or disorder within a node.

The Decision Tree hyperparameters are optimized separately for both:

* Gini
* Entropy

The resulting trees are visualized using Scikit-learn's `plot_tree()`.

The visualizations include:

* Feature used for splitting
* Split threshold
* Impurity
* Number of samples
* Class distribution
* Predicted class

---

# Results

The following results were obtained after hyperparameter optimization using 5-fold stratified cross-validation.

## Iris Dataset

| Model                | CV Accuracy | Test Accuracy |   Precision |      Recall |    F1 Score |
| -------------------- | ----------: | ------------: | ----------: | ----------: | ----------: |
| Logistic Regression  |  **96.67%** |   **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Gaussian Naive Bayes |      95.00% |        96.67% |      96.97% |      96.67% |      96.66% |
| Decision Tree        |      95.83% |        96.67% |      96.97% |      96.67% |      96.66% |
| Random Forest        |      96.67% |        96.67% |      96.97% |      96.67% |      96.66% |
| Gradient Boosting    |      96.67% |        96.67% |      96.97% |      96.67% |      96.66% |
| KNN                  |      96.67% |        93.33% |      94.44% |      93.33% |      93.27% |
| SVM                  |  **97.50%** |        93.33% |      93.33% |      93.33% |      93.33% |

### Iris Best Result

**Logistic Regression achieved 100.00% test accuracy**, with 100.00% Precision, Recall, and F1-score on the selected test split.

SVM achieved the highest cross-validation accuracy at **97.50%**, but its independent test accuracy was 93.33%.

---

## Wine Dataset

| Model                | CV Accuracy | Test Accuracy |   Precision |      Recall |    F1 Score |
| -------------------- | ----------: | ------------: | ----------: | ----------: | ----------: |
| KNN                  |      97.88% |   **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Decision Tree        |      90.25% |   **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Random Forest        |      97.91% |   **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Gradient Boosting    |      97.19% |   **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Logistic Regression  |      97.91% |        97.22% |      97.41% |      97.22% |      97.20% |
| Gaussian Naive Bayes |      97.22% |        97.22% |      97.44% |      97.22% |      97.23% |
| SVM                  |  **99.31%** |        94.44% |      95.14% |      94.44% |      94.32% |

### Wine Best Result

Four models achieved **100.00% test accuracy** on the selected test split:

* KNN
* Decision Tree
* Random Forest
* Gradient Boosting

SVM achieved the highest cross-validation accuracy at **99.31%**, but its independent test accuracy was 94.44%.

---

# Best Performing Models

| Dataset | Best Test Model(s)  | Test Accuracy |
| ------- | ------------------- | ------------: |
| Iris    | Logistic Regression |   **100.00%** |
| Wine    | KNN                 |   **100.00%** |
| Wine    | Decision Tree       |   **100.00%** |
| Wine    | Random Forest       |   **100.00%** |
| Wine    | Gradient Boosting   |   **100.00%** |

---

# Key Findings

### Iris Dataset

* Logistic Regression produced the best test-set result with **100.00% accuracy**.
* SVM achieved the highest cross-validation accuracy at **97.50%**.
* Gaussian Naive Bayes, Decision Tree, Random Forest, and Gradient Boosting achieved **96.67% test accuracy**.
* KNN achieved **93.33% test accuracy**.

### Wine Dataset

* KNN, Decision Tree, Random Forest, and Gradient Boosting achieved **100.00% test accuracy**.
* SVM achieved the highest cross-validation accuracy at **99.31%**.
* Logistic Regression and Gaussian Naive Bayes achieved **97.22% test accuracy**.

### Cross-Validation vs Test Performance

An important observation is that the model with the highest cross-validation accuracy does not necessarily achieve the highest independent test accuracy.

For example:

**Iris**

```text
SVM
CV Accuracy    = 97.50%
Test Accuracy  = 93.33%
```

**Wine**

```text
SVM
CV Accuracy    = 99.31%
Test Accuracy  = 94.44%
```

This highlights the importance of evaluating a model on data that was not used during model selection.

---

# Visualizations

The project generates several visualizations.

### Exploratory Data Analysis

* Class distribution plots
* Feature distribution plots
* Correlation heatmaps

### Model Evaluation

* Accuracy comparison
* Precision/Recall/F1 comparison
* Confusion matrices

### Model Interpretation

* Gini Decision Tree
* Entropy Decision Tree
* Random Forest feature importance

---

# Project Structure

```text
machine-learning-classification/
│
├── notebook/
│   └── ML_Classification_Benchmark.ipynb
│
├── results/
│   │
│   ├── final_model_results.csv
│   ├── grid_search_results.csv
│   ├── decision_tree_results.csv
│   │
│   ├── Iris_model_comparison.png
│   ├── Iris_metrics_comparison.png
│   ├── Iris_feature_importance.png
│   │
│   ├── Wine_model_comparison.png
│   ├── Wine_metrics_comparison.png
│   ├── Wine_feature_importance.png
│   │
│   ├── confusion_matrices/
│   │   ├── Iris_Logistic_Regression.png
│   │   ├── Iris_KNN.png
│   │   ├── Iris_Gaussian_Naive_Bayes.png
│   │   ├── Iris_SVM.png
│   │   ├── Iris_Decision_Tree.png
│   │   ├── Iris_Random_Forest.png
│   │   ├── Iris_Gradient_Boosting.png
│   │   └── ...
│   │
│   └── decision_trees/
│       ├── Iris_Decision_Tree_Gini.png
│       ├── Iris_Decision_Tree_Entropy.png
│       ├── Wine_Decision_Tree_Gini.png
│       └── Wine_Decision_Tree_Entropy.png
│
├── requirements.txt
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/machine-learning-classification.git
```

Navigate to the project directory:

```bash
cd machine-learning-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

The project requires:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

# Running the Project

## Option 1 — Kaggle

The project can be executed directly on Kaggle.

1. Open Kaggle.
2. Create a new notebook.
3. Upload/import the notebook.
4. Run the cells sequentially.
5. View the generated results and visualizations.
6. Download the generated files from `/kaggle/working/results/`.

---

## Option 2 — Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebook/ML_Classification_Benchmark.ipynb
```

Run the notebook cells sequentially.

---

# Future Improvements

The project can be extended in several ways:

* Add **XGBoost**
* Add LightGBM
* Implement Voting Classifier
* Implement Stacking Classifier
* Apply Principal Component Analysis (PCA)
* Add ROC-AUC analysis
* Generate Precision-Recall curves
* Analyze learning curves
* Add SHAP-based model interpretability
* Track experiments using MLflow
* Deploy the best model using Streamlit
* Create a REST API using Flask or FastAPI

---

# Learning Outcomes

This project demonstrates practical understanding of:

* Supervised Machine Learning
* Multiclass Classification
* Exploratory Data Analysis
* Data Preprocessing
* Feature Scaling
* Train-Test Splitting
* Stratified Cross-Validation
* Hyperparameter Optimization
* Grid Search
* Model Selection
* Naive Bayes
* KNN
* SVM
* Decision Trees
* Ensemble Learning
* Random Forest
* Gradient Boosting
* Gini Impurity
* Entropy
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Feature Importance
* Model Interpretability

---

# Conclusion

This project provides a systematic comparison of multiple machine learning classification algorithms using the Iris and Wine datasets.

Rather than relying on a single algorithm, the project evaluates seven different classification approaches and uses **stratified cross-validation and GridSearchCV** to optimize their hyperparameters.

The final evaluation demonstrates that different algorithms perform differently across datasets. Logistic Regression achieved the highest test performance on Iris, while KNN, Decision Tree, Random Forest, and Gradient Boosting achieved 100% test accuracy on the selected Wine test split.

The project also demonstrates the importance of distinguishing between **cross-validation performance and independent test performance**, providing a more reliable approach to machine learning model selection.

---

# Author

**Your Name**

* GitHub: https://github.com/KaustavJu27

---

## License

This project is intended for **educational and learning purposes**.
