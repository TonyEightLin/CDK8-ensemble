from sklearn.svm import SVC

from cdk8classifier.models.model_wrapper import ModelWrapper

MODEL_NAME = "SVC"


class SVCModel(ModelWrapper):
    def __init__(self, **kwargs):
        self._model_name = MODEL_NAME
        super().__init__(**kwargs)
        self._probability = False  # it's binary prediction
        if not self._is_production:
            super()._initialize_dataset()

    def get_pristine_model(self):
        return SVC()

    def fit_model(self):
        self._model = SVC(**self._hyperparams).fit(self._X_train, self._y_train)
        super().fit_model()
        return self
