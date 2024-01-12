# logloss function via scikit-learn

log loss function in scikit-learn. It is called `sklearn.metrics.log_loss` and it can be used to compute the log loss, aka logistic loss or cross-entropy loss, for binary or multiclass classification models¹. The log loss function takes the following parameters:

- `y_true`: array-like or label indicator matrix of the ground truth (correct) labels for n_samples samples.
- `y_pred`: array-like of float, shape = (n_samples, n_classes) or (n_samples,) of the predicted probabilities, as returned by a classifier’s predict_proba method.
- `eps`: float or “auto”, default=”auto”. Log loss is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p)). The default will depend on the data type of y_pred and is set to np.finfo(y_pred.dtype).eps.
- `normalize`: bool, default=True. If true, return the mean loss per sample. Otherwise, return the sum of the per-sample losses.
- `sample_weight`: array-like of shape (n_samples,), default=None. Sample weights.
- `labels`: array-like, default=None. If not provided, labels will be inferred from y_true.

The log loss function returns a float value that represents the log loss, aka logistic loss or cross-entropy loss. The lower the log loss, the better the model performance.

Here is an example of how to use the log loss function in Python:

```python
from sklearn.metrics import log_loss
y_true = [0, 0, 1, 1]
y_pred = [0.2, 0.8, 0.6, 0.4]
log_loss(y_true, y_pred)
```

Output: `0.7135581778200728`

Source:
(1) sklearn.metrics.log_loss — scikit-learn 1.3.0 documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html.
(2) sklearn.ensemble.RandomForestClassifier — scikit-learn 1.3.0 documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.
(3) python - How to understand the loss function in scikit-learn logestic .... https://stackoverflow.com/questions/55813551/how-to-understand-the-loss-function-in-scikit-learn-logestic-regression-code.
(4) scikit-learn - sklearn.metrics.log_loss Log loss, aka logistic or cross .... https://runebook.dev/en/docs/scikit_learn/modules/generated/sklearn.metrics.log_loss.
(5) How To Calculating Log Loss Using Scikit-learn - rasgoml.com. https://www.rasgoml.com/feature-engineering-tutorials/how-to-calculating-log-loss-using-scikit-learn.


##############################################################

# logloss be included in a gridsearchcv object, for example as a scoring method

log loss can be included in a gridsearchcv object as a scoring method. There are two ways to do this:

- You can use the string `'neg_log_loss'` as the value of the `scoring` parameter in the `GridSearchCV` constructor. This will use the negative log loss, which is the log loss multiplied by -1, as the score to optimize. The lower the log loss, the better the model performance, so the higher the negative log loss, the better the model performance. This is an example of how to use this method:

```python
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale", probability=True)
clf = GridSearchCV(svc, parameters, cv=5, scoring='neg_log_loss')
clf.fit(iris.data, iris.target)
```

- You can use a custom scorer function that wraps the `sklearn.metrics.log_loss` function and passes it to the `scoring` parameter in the `GridSearchCV` constructor. You need to use the `sklearn.metrics.make_scorer` function to create a scorer from the log loss function and set the `greater_is_better` argument to `False`, since you want to minimize the log loss. You also need to set the `needs_proba` argument to `True`, since the log loss function requires predicted probabilities as input. This is an example of how to use this method:

```python
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale", probability=True)
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
clf = GridSearchCV(svc, parameters, cv=5, scoring=LogLoss)
clf.fit(iris.data, iris.target)
```

Both methods will result in the same optimal parameters and scores for the grid search. You can find more information about how to use log loss as a scoring method in grid search in [this](^1^) and [this](^2^) web pages.

Source:
(1) How to use log_loss scorer in gridsearchcv? - Stack Overflow. https://stackoverflow.com/questions/55892224/how-to-use-log-loss-scorer-in-gridsearchcv.
(2) sklearn.model_selection.GridSearchCV - scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html.
(3) GridSearchCV with score = neg_log_loss - Stack Overflow. https://stackoverflow.com/questions/66873928/gridsearchcv-with-score-neg-log-loss.

