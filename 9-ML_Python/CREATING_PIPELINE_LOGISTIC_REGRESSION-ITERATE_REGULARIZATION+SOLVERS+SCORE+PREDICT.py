# import warnings module
import warnings

# import exceptions from sklearn, specifically ConvergenceWarning-->endless amount of ConvergenceWarning otherwise
from sklearn.exceptions import ConvergenceWarning

# filtering/ignoring the endless amount of ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Import the necessary modules
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Import the scoring methods
from sklearn.metrics import jaccard_score, f1_score, log_loss, accuracy_score

# Define the parameter grid
param_grid = {
    'solver': ['newton-cg', 'newton-cholesky', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'C': np.logspace (-2, 10, 13),
    'max_iter': [100, 500, 1000, 5000, 10000],
    'tol': [1e-4, 1e-3, 1e-2],
    'penalty': ['l1', 'l2', 'elasticnet', 'None'],
    'multi_class': ['auto', 'ovr', 'multinomial']
}

# Create a logistic regression object
log_reg = LogisticRegression()

# Create a grid search object
grid_search = GridSearchCV(log_reg, param_grid, cv=10)

# Fit the grid search on the data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Predict with the best estimator
y_pred = grid_search.predict(X_test)

# Predict probabilities with the best estimator
y_prob = grid_search.predict_proba(X_test)

# Calculate the scoring methods
jaccard = jaccard_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred)
logloss = log_loss(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)

# Print the scoring methods
print("Jaccard score: ", jaccard)
print("F1 score: ", f1)
print("Log loss: ", logloss)
print("Accuracy score: ", accuracy)
print("y_test:\n", y_test[0:5])
print("y_pred:\n",  y_pred[0:5])
print("y_prob:\n", y_prob[0:5])