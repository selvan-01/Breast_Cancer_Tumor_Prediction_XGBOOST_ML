# ===============================================
# ML Evaluation - Breast Cancer Tumor Prediction
# Comparing Multiple Machine Learning Algorithms
# ===============================================

# --------- 1. Import Required Libraries ---------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model selection & evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# --------- 2. Load Dataset ---------
# NOTE: Replace 'dataset.csv' with your dataset file

dataset = pd.read_csv('dataset.csv')

# Display dataset info
print("Dataset Shape:", dataset.shape)
print("\nFirst 5 Rows:\n", dataset.head())


# --------- 3. Separate Features (X) and Target (y) ---------
# X → input features
# y → target (tumor classification)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# --------- 4. Split Dataset into Train & Test ---------
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# --------- 5. Import Machine Learning Models ---------
# We are comparing 6 different ML algorithms

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.linear_model import LogisticRegression                  # LR
from sklearn.tree import DecisionTreeClassifier                      # Decision Tree
from sklearn.neighbors import KNeighborsClassifier                   # KNN
from sklearn.naive_bayes import GaussianNB                           # Naive Bayes
from sklearn.svm import SVC                                          # Support Vector Machine


# --------- 6. Create Model List ---------
# Each model is stored with a short name

models = []

models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# --------- 7. Evaluate Models using Cross Validation ---------
# StratifiedKFold → ensures equal class distribution

results = []   # store all accuracy scores
names = []     # store model names
mean_scores = []  # store mean accuracy

for name, model in models:
    
    # 10-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    # Evaluate model
    cv_results = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring='accuracy'
    )
    
    # Store results
    results.append(cv_results)
    names.append(name)
    mean_scores.append(cv_results.mean())
    
    # Print performance
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")


# --------- 8. Visualize Model Performance ---------
# Bar chart comparing accuracy of all models

plt.figure(figsize=(8, 5))
plt.ylim(0.5, 1.0)

plt.bar(names, mean_scores)

plt.title('ML Algorithm Comparison (Accuracy)')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')

plt.show()