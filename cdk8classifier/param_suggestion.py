import os

import numpy as np
from hyperopt import hp

from cdk8classifier import logger
from cdk8classifier.hyperparams import ParamSearch, NNParamSearch
from cdk8classifier.models import SVCModel, RandomForestModel, XGBoostModel, KNNModel, NeuralNetworkModel, \
    LogisticRegressionModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def suggest_hyperparams(model, space):
    if model.get_model_name() == 'Neural Network':
        result = NNParamSearch(model, space).run()
    else:
        result = ParamSearch(model, space).run()

    logger.log_info(msg=f"Suggested hyperparams for {model.get_model_name()}: {result}")


def suggest_lr_hp():
    space = {
        'max_iter': 1000,
        'penalty': hp.choice('penalty', ['l1', 'l2']),
        # 'C': hp.uniform('C', 0, 20),
        'solver': hp.choice('solver', ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'])
    }
    suggest_hyperparams(LogisticRegressionModel(), space)


def suggest_knn_hp():
    space = {
        # has to begin from 0 to display correct n_neighbors (the hp.choice suggestion shows index from 0)
        'n_neighbors': hp.choice('n_neighbors', range(0, 50))
    }
    suggest_hyperparams(KNNModel(), space)


def suggest_svc_hp():
    space = {
        'probability': True,  # for neg_log_loss
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20)
    }
    suggest_hyperparams(SVCModel(), space)


def suggest_rf_hp():
    space = {
        'max_depth': hp.choice('max_depth', np.arange(0, 20, dtype=int)),
        # 'max_features': hp.choice('max_features', range(1,X.shape[1]+1)),
        'n_estimators': hp.choice('n_estimators', np.arange(0, 200, dtype=int)),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    }
    suggest_hyperparams(RandomForestModel(), space)


def suggest_xgb_hp():
    space = {
        'max_depth': hp.choice('max_depth', np.arange(0, 20, dtype=int)),
        # 'n_estimators': hp.choice('n_estimators', np.arange(0, 200, dtype=int)),
        'n_estimators': 300,
        'eta': hp.quniform('eta', 0.01, 0.35, 0.01),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.35, 0.01),
        'min_child_weight': hp.choice('min_child_weight', np.arange(0, 10, dtype=int)),
        'subsample': hp.quniform('subsample', 0.7, 1.0, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1.0, 0.1),
        'objective': 'binary:logistic',
        'seed': 0,
        'use_label_encoder': False  # to avoid warning
    }
    suggest_hyperparams(XGBoostModel(), space)


def suggest_nn_hp():
    space = {
        'optimizer': hp.choice('optimizer', ['SGD', 'adam']),
        # 'optimizer': 'SGD',
        'lr': hp.quniform('lr', 0.01, 0.5, 0.02),
        'percentage': hp.quniform('percentage', 0.01, 0.99, 0.05),
        'shrink': hp.quniform('shrink', 0.01, 0.99, 0.05),
        'dropout': hp.quniform('dropout', 0, 0.699, 0.1)
    }
    suggest_hyperparams(NeuralNetworkModel(), space)


if __name__ == '__main__':
    suggest_lr_hp()
    suggest_knn_hp()
    suggest_svc_hp()
    suggest_rf_hp()
    suggest_xgb_hp()
    suggest_nn_hp()
    pass
