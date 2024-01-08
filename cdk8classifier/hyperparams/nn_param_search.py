import statistics

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, Adamax

from cdk8classifier import logger
from cdk8classifier.commons import configs
from cdk8classifier.hyperparams.param_search import ParamSearch, MAX_EVALS, skip_zero, SKIP_ZERO_LIST, CV_FOLDS
from cdk8classifier.models.neural_network import build_model

TEST_SIZE = configs['test_size']
PATIENCE = configs['early_stopping']['patience']
EPOCHS = configs['keras']['epochs']
LOG_LEVEL = configs['project_log_level']


def define_optimizer(space):
    if space['optimizer'] == 'SGD':
        return SGD(learning_rate=space['lr'])
    elif space['optimizer'] == 'adamax':
        return Adamax(learning_rate=space['lr'])
    else:
        return Adam()  # use default learning_rate
        # return Adam(learning_rate=space['lr'])


class NNParamSearch(ParamSearch):
    def __init__(self, model, space):
        super().__init__(model, space)
        self._input_dim = self._X.shape[1]

    def run(self):
        def objective(space):
            for property_name in SKIP_ZERO_LIST:
                skipped = skip_zero(property_name, space)
                if skipped:
                    return skipped

            losses = []
            epochs = []
            ind_n = 0
            optimizer = define_optimizer(space)
            skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42)
            for train_index, test_index in skf.split(self._X, self._y):
                X_train, X_test = self._X[train_index], self._X[test_index]
                y_train, y_test = self._y[train_index], self._y[test_index]

                model = build_model(
                    percentage=space['percentage'],
                    shrink=space['shrink'],
                    dropout=space['dropout'],
                    input_dim=self._input_dim,
                    optimizer=optimizer
                )
                # describe model once
                if ind_n == 0 and LOG_LEVEL == 'debug':
                    model.summary()

                monitor = EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-3,
                    patience=PATIENCE,
                    verbose=0,
                    mode='auto',
                    restore_best_weights=True
                )
                result = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    verbose=0,
                    epochs=EPOCHS,
                    callbacks=[monitor]
                )
                ind_n += 1
                epochs.append(monitor.stopped_epoch)
                losses.append(np.amin(result.history['val_loss']))

            mean_score = statistics.mean(losses)
            mean_epoch = statistics.mean(epochs)
            logger.log_trace(msg=f"mean validation loss: {mean_score}, mean epoch: {mean_epoch}, space: {space}")
            return {'loss': mean_score, 'status': STATUS_OK}

        trials = Trials()

        best_hyperparams = fmin(
            fn=objective,
            space=self._space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials
        )

        return best_hyperparams
