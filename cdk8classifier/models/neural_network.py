from pathlib import PurePosixPath

import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from cdk8classifier import logger, visualizations as vis
from cdk8classifier.commons import configs, PROJECT_ROOT_PATH
from cdk8classifier.models.model_wrapper import ModelWrapper

MODEL_NAME = "Neural Network"
ACTIVATION = configs['neural_network']['activation']
EPOCHS = configs['keras']['epochs']
PATIENCE = configs['early_stopping']['patience']
LOG_LEVEL = configs['project_log_level']
N_COMPONENTS = configs['pca']['n_components']


# For evaluation, we can also use
# score = model.evaluate(X_test, y_test, verbose=0)
# score[0] is accuracy, score[1] is loss
class NeuralNetworkModel(ModelWrapper):
    def __init__(self, **kwargs):
        self._model_name = MODEL_NAME
        super().__init__(**kwargs)
        self._probability = True  # keras predicts probability
        self._fit_result = None
        self._should_scale = True
        if not self._is_production:
            super()._initialize_dataset()
            self._input_dim = self._X_train.shape[1]
        self._repo_path = PurePosixPath(PROJECT_ROOT_PATH) / 'repo' / 'neural_network'

    def fit_model(self):
        self._model = build_model(input_dim=self._input_dim, **self._hyperparams)
        if LOG_LEVEL == 'debug':
            self._model.summary()

        monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-3,
            patience=PATIENCE,
            verbose=0,
            mode='auto',
            restore_best_weights=True
        )
        self._fit_result = self._model.fit(
            self._X_train, self._y_train,
            validation_data=(self._X_test, self._y_test),
            verbose=0,
            epochs=EPOCHS,
            callbacks=[monitor]
        )
        super().fit_model()
        return self

    def fit_pca_model(self):
        self._input_dim = N_COMPONENTS
        super().fit_pca_model()

    def save(self):
        self._model.save(self._repo_path)

    def load(self):
        self._model = load_model(self._repo_path)
        return self

    def get_proba_pred(self):
        return reshape_array(self._y_pred)

    def plt_learning_curves(self, save_pic=False):
        logger.log_trace(msg=f"history keys: {self._fit_result.history.keys()}")
        vis.plt_nn_learning_curve(self._fit_result.history, show_loss=True, save_fig=save_pic)


def define_regularizer():
    regularizer = configs['neural_network']['activity_regularizer']
    regularizer_type = regularizer['type']
    regularizer_value = float(regularizer['value'])
    if regularizer_type == 'L1':
        regularizer = regularizers.l1(regularizer_value)
    elif regularizer_type == 'L2':
        regularizer = regularizers.l2(regularizer_value)
    elif regularizer_type == 'L1_L2':
        regularizer = regularizers.l1_l2(l1=regularizer_value, l2=regularizer_value)
    else:
        regularizer = None
    return regularizer


def build_model(**kwargs):
    input_dim = kwargs['input_dim']
    percentage = kwargs['percentage']
    shrink = kwargs['shrink']
    dropout = kwargs['dropout']
    optimizer = kwargs['optimizer']

    neuron_num = int(percentage * 5000)
    layer = 0
    model = Sequential()

    while neuron_num > 25 and layer < 5:
        if layer == 0:
            model.add(Dense(
                neuron_num,
                activation=ACTIVATION,
                input_dim=input_dim,
                activity_regularizer=define_regularizer()
            ))
        else:
            model.add(Dense(neuron_num, activation=ACTIVATION))
        layer += 1
        model.add(Dropout(dropout))
        neuron_num = neuron_num * shrink

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def reshape_array(predictions):
    proba_pred = np.zeros(0)
    x_len = 0
    for probability in predictions:
        probability = np.insert(probability, 0, 1 - probability[0])
        proba_pred = np.append(proba_pred, probability)
        x_len = int(proba_pred.size / 2)  # decide shape x
    return proba_pred.reshape(x_len, 2)
