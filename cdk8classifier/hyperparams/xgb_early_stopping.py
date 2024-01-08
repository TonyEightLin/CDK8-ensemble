import xgboost as xgb

from cdk8classifier import logger
from cdk8classifier import visualizations as vis
from cdk8classifier.commons import configs
from cdk8classifier.models import XGBoostModel

N_FOLDS = configs['cv_folds']
EARLY_STOPPING_ROUNDS = configs['early_stopping']['patience']


class XGBEarlyStopping:
    def __init__(self, model):
        self._X_train, self._X_test, self._y_train, self._y_test = model.get_train_test_dataset()
        self._hyperparams = model.get_hyperparams()
        self._xgb_cv = None
        logger.log_trace(msg=f"xgb early stopping: {self._hyperparams}")

    def get_xgb_cv(self):
        return self._xgb_cv

    def run(self):
        data_dmatrix = xgb.DMatrix(data=self._X_train, label=self._y_train)
        self._xgb_cv = xgb.cv(
            dtrain=data_dmatrix,
            params=self._hyperparams,
            nfold=N_FOLDS,
            num_boost_round=500,  # self._hyperparams['n_estimators'],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            metrics=["logloss", "aucpr"],  # auc, aucpr, logloss...
            as_pandas=True,
            seed=123)
        return self._xgb_cv.tail(5)


if __name__ == '__main__':
    xgboost = XGBoostModel(hyperparams=configs['xgboost'])
    early_stopping = XGBEarlyStopping(xgboost)
    print("reference for n_estimators, check df index\n", early_stopping.run())
    vis.plt_xgb_learning_curve(early_stopping.get_xgb_cv(), save_fig=False)
    vis.plt_xgb_learning_curve(early_stopping.get_xgb_cv(), show_loss=True, save_fig=False)
