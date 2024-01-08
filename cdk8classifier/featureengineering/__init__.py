from pathlib import PurePosixPath

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from cdk8classifier.commons import utils, MODEL_DATA_FILE_PATH, TRAINING_ROOT_PATH
from cdk8classifier.featureengineering import chem_utils
from cdk8classifier.featureengineering.fingerprint_vector import FingerprintVector
from cdk8classifier.featureengineering.similarity import Similarity


def positive_rate_in_top_counts(base_num):
    sim_df = utils.to_dataframe(PurePosixPath(TRAINING_ROOT_PATH) / 'fragment-similarities.csv')
    sim_df.sort_values('Match Count', inplace=True, ascending=False)
    p_data = sim_df[sim_df['Active'] == 'p']
    top = sim_df.head(base_num)
    top_p = top[top['Active'] == 'p']
    rate = "{:.2%}".format(len(top_p) / base_num)
    print(f'Total positive count is {len(p_data)}.',
          f'Ordering by fragment match count, positive rate in top {base_num} is {rate}')


# compare if utils.scale_df works the same
def compare_scaling():
    # scale before model fitting
    X, y = utils.xy_dataset(MODEL_DATA_FILE_PATH)
    X_train, X_test, y_train, y_test = utils.train_test_dataset(X, y)

    scaler = MinMaxScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(metrics.classification_report(y_test, y_pred))

    # scale the whole df first
    scaled_df = utils.scale_df_features(utils.to_dataframe(MODEL_DATA_FILE_PATH))
    # print(scaled_df.head())
    X, y = utils.to_dataset(scaled_df)
    X_s_train, X_s_test, y_s_train, y_s_test = utils.train_test_dataset(X, y)
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X_s_train, y_s_train)
    y_pred = model.predict(X_s_test)
    print(metrics.classification_report(y_s_test, y_pred))


if __name__ == '__main__':
    for num in range(100, 400, 100):
        positive_rate_in_top_counts(num)

    # compare_scaling()
