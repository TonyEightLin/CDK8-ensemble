import os

from cdk8classifier import logger
from cdk8classifier import visualizations as vis
from cdk8classifier.commons import configs
from cdk8classifier.hyperparams import XGBEarlyStopping
from cdk8classifier.models import KNNModel, RandomForestModel, SVCModel, XGBoostModel, NeuralNetworkModel, \
    LogisticRegressionModel

# suppress tensorflow logging info
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SAVE_FIG = configs['figs']['save']
WITH_PCA = configs['pca']['apply']


def save_models(*models):
    for m in models:
        m.save()


def do_individual_plots(model):
    # model.plt_threshold_comparison(SAVE_FIG)
    model.plot_confusion_matrix(SAVE_FIG)
    # model.plt_pr_curve(SAVE_FIG)
    # model.plt_roc(SAVE_FIG)
    pass


def evaluate_lr():
    logger.log_info(msg="lr evaluation -----------------------------------------------------")
    model = LogisticRegressionModel(hyperparams=configs['lr'])
    if WITH_PCA:
        model.fit_pca_model()
    else:
        model.fit_model()
    model.evaluate()
    logger.log_info(msg="--------------------------------------------------------------------")
    do_individual_plots(model)
    return model


def evaluate_knn():
    logger.log_info(msg="knn evaluation -----------------------------------------------------")
    model = KNNModel(hyperparams=configs['knn'])
    if WITH_PCA:
        model.fit_pca_model()
    else:
        model.fit_model()
    model.evaluate()
    logger.log_info(msg="--------------------------------------------------------------------")
    do_individual_plots(model)
    return model


def evaluate_svc():
    logger.log_info(msg="SVC evaluation -----------------------------------------------------")
    model = SVCModel(hyperparams=configs['svc'])
    if WITH_PCA:
        model.fit_pca_model()
    else:
        model.fit_model()
    model.evaluate()
    logger.log_info(msg="--------------------------------------------------------------------")
    do_individual_plots(model)
    return model


def evaluate_rf():
    logger.log_info(msg="Random Forest evaluation -------------------------------------------")
    model = RandomForestModel(hyperparams=configs['random_forest'])
    if WITH_PCA:
        model.fit_pca_model()
    else:
        model.fit_model()
    model.evaluate()
    logger.log_info(msg="--------------------------------------------------------------------")
    do_individual_plots(model)
    return model


def evaluate_xgb():
    logger.log_info(msg="XGBoost evaluation -------------------------------------------------")

    model = XGBoostModel(hyperparams=configs['xgboost'])
    if WITH_PCA:
        model.fit_pca_model()
    else:
        model.fit_model()
    model.evaluate()
    logger.log_info(msg="--------------------------------------------------------------------")
    do_individual_plots(model)

    early_stopping = XGBEarlyStopping(model)
    early_stopping.run()
    cv = early_stopping.get_xgb_cv()
    vis.plt_xgb_learning_curve(cv, SAVE_FIG)
    vis.plt_xgb_learning_curve(cv, show_loss=False, save_fig=SAVE_FIG)
    return model


def evaluate_nn():
    logger.log_info(msg="Neural Network evaluation ------------------------------------------")
    model = NeuralNetworkModel(hyperparams=configs['neural_network'])
    if WITH_PCA:
        model.fit_pca_model()
    else:
        model.fit_model()
    model.evaluate()
    logger.log_info(msg="--------------------------------------------------------------------")
    do_individual_plots(model)
    model.plt_learning_curves(SAVE_FIG)
    return model


if __name__ == '__main__':
    lr = evaluate_lr()
    knn = evaluate_knn()
    svc = evaluate_svc()
    rf = evaluate_rf()
    xgb = evaluate_xgb()
    nn = evaluate_nn()

    all_models = lr, knn, svc, rf, xgb, nn
    save_models(*all_models)
    vis.plt_multiple_roc(*all_models, save_fig=SAVE_FIG)
    vis.plt_multiple_pr_curve(*all_models, save_fig=SAVE_FIG)
