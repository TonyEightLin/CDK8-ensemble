from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold

from cdk8classifier import logger
from cdk8classifier.commons import configs

CV_FOLDS = configs['cv_folds']
SCORING = configs['cv_scoring']
MAX_EVALS = configs['hyperopt_max_evals']
WITH_PCA = configs['pca']['apply']
N_COMPONENTS = configs['pca']['n_components']

SKIP_ZERO_LIST = ['n_neighbors', 'max_depth', 'n_estimators', 'percentage', 'shrink']


# avoid ValueError, e.g. ValueError: n_estimators must be greater than zero, got 0.
def skip_zero(property_name, space):
    if property_name in space and space[property_name] == 0:
        # give a higher loss penalty
        return {'loss': 10, 'status': STATUS_OK}


class ParamSearch:
    def __init__(self, model, space):
        self._model = model.get_pristine_model()
        self._X, self._y = model.get_xy_dataset()

        if WITH_PCA:
            pca = PCA(n_components=N_COMPONENTS).fit(self._X)
            self._X = pca.transform(self._X)

        self._space = space
        logger.log_trace(msg=f"cv_folds: {CV_FOLDS}, cv_scoring: {SCORING}")

    def run(self):
        def objective(space):
            for property_name in SKIP_ZERO_LIST:
                skipped = skip_zero(property_name, space)
                if skipped:
                    return skipped

            model = self._model.set_params(**space)
            # scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            # f1_macro, neg_log_loss, accuracy, roc_auc...
            # 'neg_log_loss' requires predict_proba support
            scores = cross_val_score(
                model,
                self._X, self._y,
                cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42),
                scoring=SCORING
            )
            mean_score = scores.mean()
            logger.log_trace(msg=f"cross validation mean_score: {mean_score}, space: {space}")
            return {'loss': -mean_score, 'status': STATUS_OK}

        trials = Trials()

        best_hyperparams = fmin(
            fn=objective,
            space=self._space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials
        )

        return best_hyperparams
