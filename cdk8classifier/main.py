import os
from pathlib import PurePosixPath, Path

import pandas as pd

from cdk8classifier import preprocessor, logger
from cdk8classifier.commons import utils, configs, PRODUCTION_CDK8_FILE_PATH, PRODUCTION_ROOT_PATH
from cdk8classifier.models import LogisticRegressionModel, KNNModel, NeuralNetworkModel, RandomForestModel, SVCModel, \
    XGBoostModel
from cdk8classifier.preprocessor import CDK8_SMILES_FILE, CDK8_ITERATION

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
THRESHOLD = configs['threshold']
COUNT_TO_PASS = configs['polling']['count_to_pass']
SKIP_FRAGMENTS = configs['datasource']['skip_fragments']
WITH_PCA = configs['pca']['apply']
CHUNK_SIZE = configs['datasource']['chunk_size']


def predict_and_vote():
    _models = [
        KNNModel(is_production=True),
        LogisticRegressionModel(is_production=True),
        NeuralNetworkModel(is_production=True),
        RandomForestModel(is_production=True),
        SVCModel(is_production=True),
        XGBoostModel(is_production=True)
    ]
    x_df = preprocessor.combine_dfs(is_production=True, skip_fragments=SKIP_FRAGMENTS)
    votes = pd.DataFrame()

    for _model in _models:
        class_name = type(_model).__name__
        model_on_board = configs['polling']['voters'][class_name]
        if model_on_board:
            _model.set_X_production(x_df)
            predictions = _model.load().predict(with_pca=WITH_PCA)
            votes[class_name] = predictions

    return votes


def generate_results(ballots):
    results = [sum(row.values >= THRESHOLD) >= COUNT_TO_PASS for _, row in ballots.iterrows()]
    ballots['Prediction'] = results
    ballots['Prediction'] = ballots['Prediction'].map({False: 'n', True: 'p'})
    cdk8_df = utils.to_dataframe(PRODUCTION_CDK8_FILE_PATH)
    return pd.concat([cdk8_df, ballots], axis=1)


def iterate_to_predict():
    large_data = pd.read_csv(PurePosixPath(PRODUCTION_ROOT_PATH) / CDK8_ITERATION, chunksize=CHUNK_SIZE)
    total_round = sum(1 for _ in large_data)
    logger.log_info(
        msg=f"Running prediction: cutting up source file into {CHUNK_SIZE} rows at a time, "
            f"total round is {total_round}, {COUNT_TO_PASS} votes to pass...")
    large_data = pd.read_csv(PurePosixPath(PRODUCTION_ROOT_PATH) / CDK8_ITERATION, chunksize=CHUNK_SIZE)

    for idx, chunk in enumerate(large_data):
        # for testing
        # if idx < 195:
        #     continue
        logger.log_info(msg=f"round {idx + 1}/{total_round}...")
        # delete these files before processing. or they would be treated as cache
        utils.delete_file(Path(PRODUCTION_ROOT_PATH) / 'cdk8-bits.csv')
        utils.delete_file(Path(PRODUCTION_ROOT_PATH) / 'cdk8-smiles.csv')
        utils.delete_file(Path(PRODUCTION_ROOT_PATH) / 'fragment-similarities.csv')
        utils.to_csv(chunk, CDK8_SMILES_FILE, save_to=PRODUCTION_ROOT_PATH)

        ballot_df = predict_and_vote()
        df = generate_results(ballot_df)
        # first round doesn't append, the rest does
        to_append = (idx != 0)
        utils.to_csv(df, 'cdk8-predictions.csv', save_to=PRODUCTION_ROOT_PATH, append=to_append)


# only keep the positive results
def extract_prediction():
    predictions = pd.read_csv(PurePosixPath(PRODUCTION_ROOT_PATH) / 'cdk8-predictions.csv')
    predictions = predictions[(predictions['Prediction'] == 'p')]
    logger.log_info(msg=f"saving {len(predictions)} positive records...")
    utils.to_csv(predictions, 'extracted-predictions.csv', save_to=PRODUCTION_ROOT_PATH)


if __name__ == '__main__':
    iterate_to_predict()
    extract_prediction()
    pass
