import xgboost as xgb

from cdk8classifier.commons import configs
from cdk8classifier.models.model_wrapper import ModelWrapper

MODEL_NAME = "XGBoost"
MAX_NUM_FEATURES = configs['xgb']['max_num_features']


# no need to normalize the features for xgboost
class XGBoostModel(ModelWrapper):
    def __init__(self, **kwargs):
        self._model_name = MODEL_NAME
        super().__init__(**kwargs)
        self._probability = False  # it's binary prediction
        if not self._is_production:
            super()._initialize_dataset()

    def get_pristine_model(self):
        return xgb.XGBClassifier(verbosity=0)

    def fit_model(self):
        self._model = xgb.XGBClassifier(**self._hyperparams)
        self._model.fit(self._X_train, self._y_train)
        super().fit_model()
        return self
