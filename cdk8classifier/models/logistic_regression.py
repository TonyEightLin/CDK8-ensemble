from sklearn.linear_model import LogisticRegression

from cdk8classifier.models.model_wrapper import ModelWrapper

MODEL_NAME = "Logistic Regression"


class LogisticRegressionModel(ModelWrapper):
    def __init__(self, **kwargs):
        self._model_name = MODEL_NAME
        super().__init__(**kwargs)
        self._probability = False
        self._should_scale = True
        if not self._is_production:
            super()._initialize_dataset()

    def get_pristine_model(self):
        return LogisticRegression()

    def fit_model(self):
        self._model = LogisticRegression(**self._hyperparams)
        self._model = self._model.fit(self._X_train, self._y_train)

        super().fit_model()
        return self
