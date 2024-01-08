from cdk8classifier.hyperparams.nn_param_search import NNParamSearch
from cdk8classifier.hyperparams.param_search import ParamSearch
from cdk8classifier.hyperparams.xgb_early_stopping import XGBEarlyStopping

# Which Evaluation Metric Should You Choose?
# https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
#
#
# Accuracy:
# when your problem is balanced
#
# F1:
# in every binary classification problem where you care more about the positive class
#
# ROC AUC:
# should not use it when your data is heavily imbalanced
# false positive rate for highly imbalanced datasets is pulled down due to a large number of true negatives
#
# PR AUC:
# when you want to choose the threshold that fits the business problem
# when your data is heavily imbalanced


# xgboost eval_metric
# https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
