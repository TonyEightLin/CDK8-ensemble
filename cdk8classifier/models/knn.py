from sklearn.neighbors import KNeighborsClassifier

from cdk8classifier.models.model_wrapper import ModelWrapper

MODEL_NAME = "K-Nearest Neighbors"


class KNNModel(ModelWrapper):
    def __init__(self, **kwargs):
        self._model_name = MODEL_NAME
        super().__init__(**kwargs)
        self._probability = False  # it's binary prediction
        self._should_scale = False  # seems to perform better without scaling
        if not self._is_production:
            super()._initialize_dataset()

    def get_pristine_model(self):
        return KNeighborsClassifier()

    def fit_model(self):
        self._model = KNeighborsClassifier(**self._hyperparams).fit(self._X_train, self._y_train)
        super().fit_model()
        return self
