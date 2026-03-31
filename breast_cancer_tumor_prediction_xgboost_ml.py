# ============================================
# Breast Cancer Tumor Prediction using XGBoost
# ============================================

# --------- 1. Import Required Libraries ---------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier


# --------- 2. Load Dataset ---------
# NOTE: Replace 'dataset.csv' with your actual file name
dataset = pd.read_csv('dataset.csv')

# Display dataset info
print("Dataset Shape:", dataset.shape)
print("\nFirst 5 Rows:\n", dataset.head())


# --------- 3. Separate Features (X) and Target (y) ---------
# X → all columns except last column
# y → last column (target: tumor type)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# --------- 4. Split Dataset into Train & Test ---------
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# --------- 5. Train Model using XGBoost ---------
# XGBClassifier is powerful for classification problems

classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
classifier.fit(X_train, y_train)


# --------- 6. Make Predictions ---------
y_pred = classifier.predict(X_test)


# --------- 7. Evaluate Model ---------
# Confusion Matrix → shows correct & incorrect predictions
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Accuracy Score → overall performance
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))


# --------- 8. K-Fold Cross Validation ---------
# More reliable performance evaluation

accuracies = cross_val_score(
    estimator=classifier,
    X=X_train,
    y=y_train,
    cv=10
)

print("\nCross Validation Accuracy: {:.2f}%".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f}%".format(accuracies.std() * 100))