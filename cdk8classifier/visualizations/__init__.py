from pathlib import PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, PrecisionRecallDisplay, \
    RocCurveDisplay

from cdk8classifier.commons import configs, utils

THRESHOLDS = np.linspace(0, 1, 20, endpoint=True)
FIGURE_LOCATION = configs['figs']['save_to']
DPI = configs['figs']['dpi']


def get_scores_by_threshold(y_test, probability_pred, thresholds):
    # precision-recall curve
    _pr_list = []
    # f1, combines precision and recall
    _f1_list = []
    # false negative rate
    _fnr_list = []

    for _threshold in thresholds:
        y_pred = (probability_pred[:, 1] >= _threshold)
        _pr_list.append(average_precision_score(y_test, y_pred))
        _f1_list.append(f1_score(y_test, y_pred, average='macro'))
        cm = confusion_matrix(y_test, y_pred)
        # TN = cm[0][0]
        # FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]
        FNR = FN / (TP + FN)
        _fnr_list.append(FNR)

    return _pr_list, _f1_list, _fnr_list


def plt_scores_by_threshold(model_name, pr_list, f1_list, fnr_list, save_fig=False):
    plt.plot(THRESHOLDS, pr_list, 'g', label='PR AUC')
    plt.plot(THRESHOLDS, f1_list, 'b', label='F1')
    plt.plot(THRESHOLDS, fnr_list, 'r', label='False Negative Rate')
    plt.title(f"Threshold Comparison - {model_name}")
    plt.legend()
    plt.ylabel('Rate')
    plt.xlabel(f'Threshold')
    if save_fig:
        save_pic('threshold', model_name)
    plt.show()


def plt_confusion_matrix(model_name, y_test, y_predicted, save_fig=False):
    sns.set()
    cm = confusion_matrix(y_test, y_predicted)
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.xaxis.set_ticklabels(['N', 'P'])
    ax.yaxis.set_ticklabels(['N', 'P'])

    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    if save_fig:
        save_pic('cm', model_name)
    plt.show()


def plt_roc(model_name, y_test, y_predicted, save_fig=False):
    display = RocCurveDisplay.from_predictions(y_test, y_predicted, name=model_name)
    _ = display.ax_.set_title(f"ROC - {model_name}")
    if save_fig:
        save_pic('roc', model_name)
    plt.show()


# AP (average_precision_score): Average Precision is a single number used to summarise a Precision-Recall curve
def plt_pr_curve(model_name, y_test, y_predicted, save_fig=False):
    display = PrecisionRecallDisplay.from_predictions(y_test, y_predicted, name=model_name)
    _ = display.ax_.set_title(f"Precision-Recall Curve - {model_name}")
    if save_fig:
        save_pic('pr-curve', model_name)
    plt.show()


def plt_multiple_roc(*models, save_fig=False):
    roc_display = None
    for idx, model in enumerate(models):
        y_test = model.get_y_test()
        y_pred = model.get_positive_proba_pred()
        name = model.get_model_name()
        if idx == 0:
            roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, name=name)
        else:
            RocCurveDisplay.from_predictions(y_test, y_pred, name=name, ax=roc_display.ax_)
    roc_display.ax_.set_title("ROC (Receiver Operating Characteristic)")
    if save_fig:
        save_pic('roc', 'multiple')
    plt.show()


def plt_multiple_pr_curve(*models, save_fig=False):
    pr_display = None
    for idx, model in enumerate(models):
        y_test = model.get_y_test()
        y_pred = model.get_positive_proba_pred()
        name = model.get_model_name()
        if idx == 0:
            pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, name=name)
        else:
            PrecisionRecallDisplay.from_predictions(y_test, y_pred, name=name, ax=pr_display.ax_)
    pr_display.ax_.set_title("Precision-Recall Curve")
    if save_fig:
        save_pic('pr-curve', 'multiple')
    plt.show()


def plt_xgb_learning_curve(xgb_cv, show_loss=True, save_fig=False):
    if show_loss:
        plt.plot(xgb_cv['train-logloss-mean'])
        plt.plot(xgb_cv['test-logloss-mean'])
        plt.ylabel('Log Loss')
        plt.legend(['train', 'test'], loc='upper right')
    else:
        plt.plot(xgb_cv['train-aucpr-mean'])
        plt.plot(xgb_cv['test-aucpr-mean'])
        plt.ylabel('PR AUC')
        plt.legend(['train', 'test'], loc='lower right')

    plt.title('Learning Curve - XGBoost')
    plt.xlabel('n_estimators')
    if save_fig:
        save_pic('learning-curve', 'xgboost')
    plt.show()


def plt_nn_learning_curve(history, show_loss=True, save_fig=False):
    if show_loss:
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.ylabel('Loss')
        plt.legend(['train', 'test'], loc='upper right')
    else:
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.ylabel('Accuracy')
        plt.legend(['train', 'test'], loc='lower right')

    plt.title('Learning Curve - Neural Network')
    plt.xlabel('Epoch')
    if save_fig:
        save_pic('learning-curve', 'Neural Network')
    plt.show()


def plt_pca_explained_variance(pca, max_n):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel(f'Number of components (max={max_n})')
    plt.ylabel('Cumulative explained variance')
    plt.show()


def plt_pca_tsne(df, pca, pca_components, perplexity, save_fig=False):
    palette = sns.color_palette("hls", 2)
    y_label = "active"
    plt.figure(figsize=(16, 4))

    pca_variance_ratio = pca.get_explained_variance_ratio()
    ratio = "{:.2%}".format(pca_variance_ratio)

    ax1 = plt.subplot(1, 3, 1)
    plt.title(f"PCA (n_components={pca_components}, ratio={ratio})")
    sns.scatterplot(
        x="pca-1", y="pca-2",
        hue=y_label,
        palette=palette,
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax1
    )

    ax2 = plt.subplot(1, 3, 2)
    plt.title(f"tSNE (perplexity={perplexity})")
    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue=y_label,
        palette=palette,
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax2
    )

    ax3 = plt.subplot(1, 3, 3)
    plt.title("tSNE transformed from PCA")
    sns.scatterplot(
        x="tsne-pca-1", y="tsne-pca-2",
        hue=y_label,
        palette=palette,
        data=df,
        legend="full",
        alpha=0.5,
        ax=ax3
    )

    if save_fig:
        save_pic('pca', 't-SNE')
    plt.show()


def plt_xgboost_feature_selection(xgb, ml_model, save_fig=False):
    max_num = configs['xgb']['max_num_features']
    ml_model.get_booster().feature_names = utils.feature_name_list()
    _, ax = plt.subplots(figsize=(10, 9))
    xgb.plot_importance(ml_model, max_num_features=max_num, ax=ax)
    plt.tick_params(axis='y', labelsize=8)
    if save_fig:
        save_pic('feature-importance', 'xgboost')
    plt.show()

    feature_importance = ml_model.get_booster().get_score(importance_type='weight')
    sorted_fi = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    features = [k for k, v in sorted_fi]
    print(f"feature importance list: {features[:100]}")


def save_pic(topic, model_name):
    file_name = utils.to_dash_lower(f'{topic}-{model_name}.png')
    file = PurePosixPath(FIGURE_LOCATION) / file_name
    plt.savefig(file, dpi=DPI)


def criteria_chart():
    df = pd.DataFrame([
        ['With Fragments / Without PCA', 'LogisticRegression', 0.97],
        ['With Fragments / Without PCA', 'KNN', 0.94],
        ['With Fragments / Without PCA', 'SVC', 1.00],
        ['With Fragments / Without PCA', 'RandomForest', 0.97],
        ['With Fragments / Without PCA', 'XGBoost', 0.97],
        ['With Fragments / Without PCA', 'NeuralNetwork', 0.97],

        ['Without Fragments / Without PCA', 'LogisticRegression', 0.97],
        ['Without Fragments / Without PCA', 'KNN', 0.94],
        ['Without Fragments / Without PCA', 'SVC', 1.00],
        ['Without Fragments / Without PCA', 'RandomForest', 1.00],
        ['Without Fragments / Without PCA', 'XGBoost', 0.97],
        ['Without Fragments / Without PCA', 'NeuralNetwork', 0.97],

        ['With Fragments / With PCA', 'LogisticRegression', 0.94],
        ['With Fragments / With PCA', 'KNN', 0.94],
        ['With Fragments / With PCA', 'SVC', 1.00],
        ['With Fragments / With PCA', 'RandomForest', 1.00],
        ['With Fragments / With PCA', 'XGBoost', 1.00],
        ['With Fragments / With PCA', 'NeuralNetwork', 1.00],

        ['Without Fragments / With PCA', 'LogisticRegression', 0.97],
        ['Without Fragments / With PCA', 'KNN', 1.00],
        ['Without Fragments / With PCA', 'SVC', 1.00],
        ['Without Fragments / With PCA', 'RandomForest', 1.00],
        ['Without Fragments / With PCA', 'XGBoost', 1.00],
        ['Without Fragments / With PCA', 'NeuralNetwork', 1.00],
    ], columns=['Criteria', 'Model', 'Scores'])
    plt.figure(figsize=(20, 8))
    ax = sns.barplot(x='Criteria', y='Scores', hue='Model', data=df)
    ax.set_title('Performance - Precision Scores: TP/ (TP + FP)')
    plt.show()


def false_positive_chart():
    df = pd.DataFrame([
        ['With Fragments / Without PCA', 'LogisticRegression', 11.67],
        ['With Fragments / Without PCA', 'KNN', 5.27],
        ['With Fragments / Without PCA', 'SVC', 1.91],
        ['With Fragments / Without PCA', 'RandomForest', 3.14],
        ['With Fragments / Without PCA', 'XGBoost', 2.69],
        ['With Fragments / Without PCA', 'NeuralNetwork', 23.91],
        ['With Fragments / Without PCA', 'Final Prediction', 2.24],

        ['Without Fragments / Without PCA', 'LogisticRegression', 10.89],
        ['Without Fragments / Without PCA', 'KNN', 4.38],
        ['Without Fragments / Without PCA', 'SVC', 0.11],
        ['Without Fragments / Without PCA', 'RandomForest', 0.00],
        ['Without Fragments / Without PCA', 'XGBoost', 3.59],
        ['Without Fragments / Without PCA', 'NeuralNetwork', 10.21],
        ['Without Fragments / Without PCA', 'Final Prediction', 2.36],

        ['With Fragments / With PCA', 'LogisticRegression', 16.50],
        ['With Fragments / With PCA', 'KNN', 1.80],
        ['With Fragments / With PCA', 'SVC', 0.00],
        ['With Fragments / With PCA', 'RandomForest', 0.79],
        ['With Fragments / With PCA', 'XGBoost', 5.05],
        ['With Fragments / With PCA', 'NeuralNetwork', 9.65],
        ['With Fragments / With PCA', 'Final Prediction', 1.35],

        ['Without Fragments / With PCA', 'LogisticRegression', 8.19],
        ['Without Fragments / With PCA', 'KNN', 0.56],
        ['Without Fragments / With PCA', 'SVC', 0.00],
        ['Without Fragments / With PCA', 'RandomForest', 4.04],
        ['Without Fragments / With PCA', 'XGBoost', 7.18],
        ['Without Fragments / With PCA', 'NeuralNetwork', 6.96],
        ['Without Fragments / With PCA', 'Final Prediction', 3.70],
    ], columns=['Groups', 'Model', 'Rate (%)'])
    plt.figure(figsize=(20, 8))
    ax = sns.barplot(x='Groups', y='Rate (%)', hue='Model', data=df)
    ax.set_title('Performance - ACD990 False Positive Rate')
    plt.show()


if __name__ == '__main__':
    criteria_chart()
    false_positive_chart()
