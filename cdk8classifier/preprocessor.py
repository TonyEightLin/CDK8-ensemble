from pathlib import Path, PurePosixPath

import pandas as pd

from cdk8classifier import logger
from cdk8classifier.commons import utils, configs, MODEL_DATA_FILE_PATH, TRAINING_CDK8_FILE_PATH, FRAG_FILE_PATH, \
    TRAINING_ROOT_PATH, PRODUCTION_ROOT_PATH, PRODUCTION_CDK8_FILE_PATH, PROJECT_ROOT_PATH
from cdk8classifier.commons.exceptions import RDKitSmilesParseError
from cdk8classifier.featureengineering import FingerprintVector, Similarity, chem_utils

ACTIVE = configs['y_label']
CDK8_SMILES_NAME = configs['cdk8']['smiles_name']
FRAG_SMILES_NAME = configs['fragments']['smiles_name']
FRAG_KEY_NAME = configs['fragments']['key_name']

CDK8_SMILES_FILE = "cdk8-smiles.csv"
CDK8_BIT_FILE = "cdk8-bits.csv"
FRAG_SIMILARITY_FILE = "fragment-similarities.csv"
MODEL_DATA_FILE = "model-data.csv"
CDK8_ITERATION = "iteration.csv"

SKIP_FRAGMENTS = configs['datasource']['skip_fragments']
WITH_PCA = configs['pca']['apply']


# do this after MODEL_DATA_FILE generated
def check_class_ratio():
    df = utils.to_dataframe(MODEL_DATA_FILE_PATH)
    ratio = "{:.2%}".format(utils.get_class_ratio(df))
    logger.log_info(msg=f"data class ratio: active rate is {ratio}")
    logger.log_info(msg=f"df shape: {df.shape}")
    logger.log_info(msg=f'SKIP_FRAGMENTS: {SKIP_FRAGMENTS}, WITH_PCA: {WITH_PCA}')


# add extra thermo cdk8 to data
def add_thermo_data():
    cdk8_df = utils.to_dataframe(TRAINING_CDK8_FILE_PATH)
    thermo_df = utils.to_dataframe(PurePosixPath(PROJECT_ROOT_PATH) / 'resources' / 'thermo.csv')
    thermo_df.drop(['Name'], axis=1, inplace=True)
    thermo_df[ACTIVE] = 'n'
    logger.log_trace(msg=f"cdk8_df shape: {cdk8_df.shape}")
    logger.log_trace(msg=f"thermo_df shape: {thermo_df.shape}")
    cdk8_df = pd.concat([cdk8_df, thermo_df], axis=0).fillna(0)
    cdk8_df.reset_index()
    utils.to_csv(cdk8_df, CDK8_SMILES_FILE, save_to=TRAINING_ROOT_PATH)


# remove duplicated data
def purge_file():
    utils.purge_csv_on_smiles(CDK8_SMILES_FILE)
    # validate
    cdk8_df = utils.to_dataframe(TRAINING_CDK8_FILE_PATH)
    du = utils.find_duplicated_rows(cdk8_df)
    logger.log_trace(msg=f"duplicated df check: {du.values}")


# combine bit vector data and similarity data
def combine_dfs(save_csv=True, is_production=False, skip_fragments=False):
    if is_production:
        cdk8_df = utils.to_dataframe(PRODUCTION_CDK8_FILE_PATH)
        root_path = PRODUCTION_ROOT_PATH
    else:
        cdk8_df = utils.to_dataframe(TRAINING_CDK8_FILE_PATH)
        root_path = TRAINING_ROOT_PATH

    bit_path = PurePosixPath(root_path) / CDK8_BIT_FILE
    sim_path = PurePosixPath(root_path) / FRAG_SIMILARITY_FILE

    frag_df = utils.to_dataframe(FRAG_FILE_PATH)
    bit_file_path = Path(utils.get_project_root()) / bit_path
    sim_file_path = Path(utils.get_project_root()) / sim_path

    logger.log_trace(
        msg=f"is_production: {is_production}, bit_file_path: {bit_file_path}, sim_file_path: {sim_file_path}")

    # if the file path exists, get dataframe from the file, otherwise generate df from scratch (and save it)
    if bit_file_path.exists():
        _bit_df = utils.to_dataframe(file_path=bit_path)
    else:
        _bit_df = FingerprintVector(
            is_production=is_production,
            molecule_smiles=cdk8_df[CDK8_SMILES_NAME],
            molecule_y_labels=cdk8_df[ACTIVE] if ACTIVE in cdk8_df else None
        ).generate_df()
    logger.log_trace(msg=f"_bit_df: {_bit_df.shape}")

    _sim_df = None
    if not skip_fragments:
        if sim_file_path.exists():
            _sim_df = utils.to_dataframe(file_path=sim_path)
        else:
            _sim_df = Similarity(
                is_production=is_production,
                molecule_smiles=cdk8_df[CDK8_SMILES_NAME],
                molecule_y_labels=cdk8_df[ACTIVE] if ACTIVE in cdk8_df else None,
                substructure_smiles=frag_df[FRAG_SMILES_NAME],
                substructure_columns=frag_df[FRAG_KEY_NAME]
            ).generate_df()
        logger.log_trace(msg=f"_sim_df: {_sim_df.shape}")

    if not is_production and not skip_fragments:
        _bit_df = _bit_df.iloc[:, :-1]  # drop the last active column

    if skip_fragments:
        df = _bit_df
    else:
        df = pd.concat([_bit_df, _sim_df], axis=1)
    df.reset_index()

    # validate
    _validate_df_shape(cdk8_df, frag_df, df, is_production=is_production, skip_fragments=SKIP_FRAGMENTS)
    if save_csv:
        if is_production:
            # handle_production_output
            pass
        else:
            utils.to_csv(df, MODEL_DATA_FILE, save_to=TRAINING_ROOT_PATH)

    return df


def _validate_df_shape(df1, df2, df3, is_production, skip_fragments=False):
    cdk8_df_shape = df1.shape
    frag_df_shape = df2.shape
    new_df_shape = df3.shape
    logger.log_trace(
        msg=f'cdk8_df_shape: {cdk8_df_shape}, frag_df_shape: {frag_df_shape}, new_df_shape: {new_df_shape}')
    assert new_df_shape[0] == cdk8_df_shape[0]
    if skip_fragments:
        if is_production:
            assert new_df_shape[1] == 1024
        else:
            assert new_df_shape[1] == 1024 + 1
    else:
        if is_production:
            assert new_df_shape[1] == 1024 + int(frag_df_shape[0]) + 1  # no Active
        else:
            assert new_df_shape[1] == 1024 + int(frag_df_shape[0]) + 2  # Active and Match Count


# original file is separated by space, here we change it to the file format which this project accepts
def convert_smi_file(is_testing=True):
    file_name = 'acd990.csv' if is_testing else 'smi.csv'
    smi_file = PurePosixPath(PROJECT_ROOT_PATH) / 'resources' / file_name
    df = pd.read_csv(smi_file, delim_whitespace=True)
    working_df = purge_smiles(df)
    working_df.columns = ['Std_Smiles', 'Name']
    utils.to_csv(working_df, CDK8_ITERATION, save_to=PRODUCTION_ROOT_PATH)


# remove those SMILES which RDKit can't parse
def purge_smiles(df):
    for index, row in df.iterrows():
        name = row[1]  # name is the second column
        try:
            chem_utils.smiles_to_fingerprint(row[0])
        except RDKitSmilesParseError as error:
            df.drop(index, inplace=True)
            logger.log_error(msg=f'SMILES of {name}: {error}')
    return df


def reduce_training_features():
    selected_features = ['bitvector157', 'bitvector43', 'bitvector64', 'bitvector15', 'Fragment 265', 'bitvector491', 'Fragment 16', 'Fragment 49', 'bitvector138', 'Fragment 227', 'bitvector32', 'Fragment 14', 'bitvector1', 'bitvector36', 'bitvector511', 'bitvector2', 'bitvector3', 'Fragment 6', 'Fragment 68', 'Fragment 33', 'Fragment 141', 'bitvector4', 'bitvector8', 'bitvector162', 'bitvector193', 'Fragment 7', 'Fragment 10', 'Fragment 11', 'Fragment 155', 'Fragment 220', 'bitvector91', 'bitvector234', 'Fragment 1', 'bitvector10', 'bitvector14', 'Fragment 130', 'Fragment 222', 'bitvector47', 'Fragment 135', 'Fragment 186', 'bitvector33', 'bitvector177', 'bitvector440', 'bitvector860', 'Fragment 15', 'Fragment 46', 'Fragment 59', 'bitvector46', 'bitvector73', 'bitvector831', 'Fragment 217', 'bitvector9', 'bitvector19', 'bitvector45', 'bitvector115', 'bitvector128', 'bitvector184', 'bitvector428', 'Fragment 40', 'Fragment 119', 'Fragment 121', 'Fragment 146', 'Fragment 176', 'Fragment 178', 'Fragment 199', 'bitvector7', 'bitvector74', 'bitvector378', 'Fragment 18', 'Fragment 29', 'Fragment 39', 'Fragment 64', 'Fragment 142', 'Fragment 188', 'Fragment 191', 'bitvector31', 'bitvector53', 'bitvector75', 'Fragment 30', 'Fragment 32', 'Fragment 36', 'Fragment 37', 'Fragment 47', 'Fragment 71', 'Fragment 114', 'Fragment 115', 'bitvector11', 'bitvector16', 'bitvector23', 'bitvector81', 'bitvector404', 'bitvector763', 'bitvector967', 'Fragment 27', 'Fragment 42', 'Fragment 65', 'Fragment 67', 'Fragment 72', 'Fragment 83', 'Fragment 88']
    df = utils.to_dataframe(MODEL_DATA_FILE_PATH)
    short_df = df[selected_features]
    short_df[ACTIVE] = df[ACTIVE]
    utils.to_csv(short_df, MODEL_DATA_FILE, save_to=TRAINING_ROOT_PATH)


if __name__ == '__main__':
    # add extra thermo cdk8 to training data
    add_thermo_data()
    # remove duplicated training data
    purge_file()
    # handle training data, can be with or without fragment data
    combine_dfs(skip_fragments=SKIP_FRAGMENTS)
    # display class ratio of training data
    check_class_ratio()

    # # is_testing=False: convert input smi file (smi.csv) into cdk8-smiles.csv for production use
    # # is_testing=True: convert acd990.csv into cdk8-smiles.csv for testing the false positive rate
    # convert_smi_file(is_testing=True)

    # # reduce features
    # reduce_training_features()
