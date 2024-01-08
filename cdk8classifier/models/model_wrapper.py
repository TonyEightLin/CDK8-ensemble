import pickle
from pathlib import PurePosixPath

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA

from cdk8classifier import logger, visualizations as vis
from cdk8classifier.commons import utils, configs, PROJECT_ROOT_PATH
from cdk8classifier.commons.exceptions import ProductionNotAppliedError

N_COMPONENTS = configs['pca']['n_components']
THRESHOLD = configs['threshold']


class ModelWrapper(object):
    def __init__(self, **kwargs):
        self._hyperparams = kwargs['hyperparams'] if 'hyperparams' in kwargs else None
        self._X_train, self._X_test, self._y_train, self._y_test = None, None, None, None
        self._model = None
        self._probability = False
        self._y_pred = None
        self._y_pred_binary = None
        self._y_pred_proba = None
        self._should_scale = False
        self._is_production = kwargs['is_production'] if 'is_production' in kwargs else False
        self._X_production = None
        self._repo_path = PurePosixPath(PROJECT_ROOT_PATH) / 'repo' / utils.to_snake_lower(f'{self._model_name}_pkl')
        self._pca_repo_path = PurePosixPath(PROJECT_ROOT_PATH) / 'repo' / 'pca_pkl'
        if not self._is_production:
            logger.log_trace(msg=f"hyper-params for {self._model_name}: {self._hyperparams}")

    # set training data, meanwhile scale it if necessary
    def _initialize_dataset(self):
        if self._is_production:
            raise ProductionNotAppliedError('_initialize_dataset')

        X, y = utils.xy_dataset(scale=True) if self._should_scale else utils.xy_dataset(scale=False)
        self._X_train, self._X_test, self._y_train, self._y_test = utils.train_test_dataset(X, y)
        logger.log_trace(
            msg=f"training data length: {len(self._X_train) + len(self._X_test)}, "
                f"feature length: {len(self._X_train[0])}, "
                f"need to scale? {self._should_scale}, check last 3 features: {self._X_train[0, -3:]}")

    # set prediction data for production, meanwhile scale it if necessary
    def set_X_production(self, x_df):
        if self._should_scale:
            self._X_production = utils.scale_df_features(x_df, is_production=True)
        else:
            self._X_production = x_df.copy()
        logger.log_trace(
            msg=f"{self.get_model_name()} needs to scale? {self._should_scale}, "
                f"check last 3 features: \n{self._X_production.iloc[0, -3:]}")

    # ml model without any params set
    def get_pristine_model(self):
        pass

    def get_hyperparams(self):
        return self._hyperparams

    def get_xy_dataset(self):
        x = np.append(self._X_train, self._X_test, axis=0)
        y = np.append(self._y_train, self._y_test, axis=0)
        return x, y

    def get_train_test_dataset(self):
        return self._X_train, self._X_test, self._y_train, self._y_test

    # model name for display
    def get_model_name(self):
        return self._model_name

    def get_y_test(self):
        return self._y_test

    def get_prediction(self):
        return self._y_pred

    # probability prediction
    def get_proba_pred(self):
        return self._y_pred_proba

    def get_positive_proba_pred(self):
        try:
            # use probability prediction and keep probabilities for the positive outcome only
            y_pred_proba = self._y_pred_proba[:, 1]
        except IndexError:
            # nn only has on proba value, instead of 2 for each class
            y_pred_proba = self._y_pred_proba
        return y_pred_proba

    # the real ml model wrapped in this class
    def get_ml_model(self):
        return self._model

    # where the model is saved
    def get_repo_path(self):
        return self._repo_path

    # does this ml model perform better with normalized/scaled data
    def should_scale(self):
        return self.should_scale()

    # abstract, should be overridden
    def fit_model(self):
        if self._is_production:
            raise ProductionNotAppliedError('fit_model')

        logger.log_trace(msg=f"fitting {self._model_name}...")
        return self

    def fit_pca_model(self):
        if self._is_production:
            raise ProductionNotAppliedError('fit_pca_model')

        pca = self.load_pca()
        if not pca:
            X, _ = self.get_xy_dataset()
            pca = PCA(n_components=N_COMPONENTS).fit(X)
            self.save_pca(pca)

        self._X_train = pca.transform(self._X_train)
        self._X_test = pca.transform(self._X_test)
        logger.log_trace(msg=f"fitting {self._model_name} with PCA...")
        return self.fit_model()

    def predict(self, with_pca=False):
        if self._is_production:  # get loaded model
            model = self._model if self._model else self.load()
            if with_pca:
                pca = self.load_pca()
                # transform data for later usage
                self._X_production = pca.transform(self._X_production)
            x_pred = self._X_production
        else:  # get fitted model
            model = self._model
            if not model:
                if with_pca:  # if fitted with pca
                    model = self.fit_pca_model()
                else:
                    model = self.fit_model()
            x_pred = self._X_test

        self._y_pred = model.predict(x_pred)

        # _y_pred_binary and _y_pred_proba is not required in predict(),
        # but we take care of it as well
        if self._probability:
            # for nn case, y_pred changes from probability to binary for evaluation
            self._y_pred_binary = self._y_pred > THRESHOLD
            self._y_pred_proba = self._y_pred
        else:
            self._y_pred_binary = self._y_pred
            if self._is_production:
                self._y_pred_proba = self._model.predict_proba(self._X_production)
            else:
                self._y_pred_proba = self._model.predict_proba(self._X_test)

        logger.log_trace(
            msg=f"{self._model_name}: "
                f"\n_y_pred_binary: \n{self._y_pred_binary}"
                f"\n_y_pred_proba: \n{self._y_pred_proba}")
        return self._y_pred

    def evaluate(self):
        if self._is_production:
            # evaluate only applies for training data
            raise ProductionNotAppliedError('evaluate')

        # if _y_pred is None, execute predict() to obtain
        y_pred = self._y_pred if self._y_pred else self.predict()
        y_pred_binary = self._y_pred_binary
        y_pred_proba = self.get_positive_proba_pred()
        y_test = self._y_test

        logger.log_trace(msg=f"predictions:\n{y_pred}")
        logger.log_trace(msg=f"confusion matrix:\n{metrics.confusion_matrix(y_test, y_pred_binary)}")
        logger.log_info(
            msg=f"classification report:\n {metrics.classification_report(y_test, y_pred_proba > THRESHOLD)}")
        logger.log_info(msg=f"accuracy: {metrics.accuracy_score(y_test, y_pred_binary):.5f}")
        logger.log_info(msg=f"average_precision_score: {metrics.average_precision_score(y_test, y_pred_proba):.5f}")
        logger.log_info(msg=f"precision_score: {metrics.precision_score(y_test, y_pred_binary):.5f}")
        logger.log_info(msg=f"roc_auc: {metrics.roc_auc_score(y_test, y_pred_proba):.5f}")

    # pickle the model
    def save(self):
        if self._is_production:
            raise ProductionNotAppliedError('save')

        logger.log_trace(msg=f'saving {self._model_name}')
        with open(self._repo_path, 'wb') as files:
            pickle.dump(self._model, files)

    # pickle pca
    def save_pca(self, pca):
        if self._is_production:
            raise ProductionNotAppliedError('save')

        logger.log_trace(msg=f'saving pca')
        with open(self._pca_repo_path, 'wb') as files:
            pickle.dump(pca, files)

    # return model wrapper
    def load(self):
        with open(self._repo_path, 'rb') as f:
            self._model = pickle.load(f)
        return self

    # return PCA
    def load_pca(self):
        try:
            with open(self._pca_repo_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def plot_confusion_matrix(self, save_pic=False):
        if self._is_production:
            raise ProductionNotAppliedError('plot_confusion_matrix')

        # if _y_pred_binary is None, execute predict() to obtain
        if self._y_pred_binary is None:
            self.predict()
        vis.plt_confusion_matrix(self._model_name, self._y_test, self._y_pred_binary, save_pic)

    def plt_threshold_comparison(self, save_pic=False):
        if self._is_production:
            raise ProductionNotAppliedError('plt_threshold_comparison')

        proba_pred = self.get_proba_pred()
        pr_list, f1_list, fnr_list = vis.get_scores_by_threshold(self._y_test, proba_pred, vis.THRESHOLDS)
        vis.plt_scores_by_threshold(self._model_name, pr_list, f1_list, fnr_list, save_pic)

    def plt_pr_curve(self, save_pic=False):
        if self._is_production:
            raise ProductionNotAppliedError('plt_pr_curve')

        vis.plt_pr_curve(self._model_name, self._y_test, self.get_positive_proba_pred(), save_pic)

    def plt_roc(self, save_pic=False):
        if self._is_production:
            raise ProductionNotAppliedError('plt_roc')

        vis.plt_roc(self._model_name, self._y_test, self.get_positive_proba_pred(), save_pic)
