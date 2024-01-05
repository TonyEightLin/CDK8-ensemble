import pathlib
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from cdk8classifier.commons.exceptions import DataNotFoundError

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)


def get_project_root() -> Path:
    # 3 levels up to root
    return Path(__file__).parent.parent.parent


def get_config_dic():
    file = PurePosixPath(_root) / 'configs.yml'
    with open(file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc


_root = get_project_root()
_configs = get_config_dic()

MODEL_DATA_FILE_PATH = _configs['datasource']['training']['model_data']


def xy_dataset(data_source=None, scale=False) -> tuple:
    _data_source = MODEL_DATA_FILE_PATH if data_source is None else data_source
    df = to_dataframe(_data_source)
    return to_dataset(df, scale)


def train_test_dataset(X, y):
    return split_dataset(X, y)


def to_dataframe(file_path):
    file = PurePosixPath(_root) / file_path
    return pd.read_csv(file)


def map_label(df):
    y_label = _configs['y_label']
    return df[y_label].map({'n': 0, 'p': 1})


def get_class_ratio(df):
    label = map_label(df)
    return sum(label) / len(label)


def to_dataset(df, scale=False):
    if scale:
        df = scale_df_features(df)

    X = df.iloc[:, :-1].values
    y = map_label(df).values

    return X, y


def split_dataset(X, y):
    return train_test_split(X, y, test_size=_configs['test_size'], stratify=y, random_state=42)


def feature_name_list(data_source=None):
    _data_source = MODEL_DATA_FILE_PATH if data_source is None else data_source
    df = to_dataframe(_data_source)
    y_label = _configs['y_label']
    return df.drop([y_label], axis=1).columns.tolist()


def to_csv(df, file_name, save_to: str = None, append: bool = False):
    if save_to is None:
        _path = PurePosixPath(_root) / 'resources'
    else:
        _path = save_to
    if append:
        df.to_csv(PurePosixPath(_path) / file_name, index=False, mode='a', header=False)
    else:
        df.to_csv(PurePosixPath(_path) / file_name, index=False)


def find_duplicated_rows(df):
    # or pd.concat(g for _, g in df.groupby(column_name) if len(g) > 1)
    return df[df.duplicated(keep=False)]


def compact_df(df, column_name):
    return df.drop_duplicates(subset=[column_name])


def scale_list_value(list_values):
    return [float(i) / max(list_values) for i in list_values]


# remove duplicated smiles rows
def purge_csv_on_smiles(file_name):
    _path = PurePosixPath('resources') / 'training' / file_name
    try:
        df = to_dataframe(_path)
    except FileNotFoundError as err:
        raise DataNotFoundError(err)

    df = compact_df(df, 'Std_Smiles')
    to_csv(df, PurePosixPath(_root) / _path)


def scale_df_features(unscaled_df, is_production=False) -> pd.DataFrame:
    _df = unscaled_df.copy()
    if is_production:
        x_cols = unscaled_df.columns[:]
    else:
        x_cols = unscaled_df.columns[:-1]
    _df[x_cols] = MinMaxScaler().fit_transform(_df[x_cols])
    return _df


def to_dash_lower(string):
    return string.replace(' ', '-').lower()


def to_snake_lower(string):
    return string.replace(' ', '_').lower()


def delete_file(filename):
    file_to_del = pathlib.Path(filename)
    file_to_del.unlink(missing_ok=True)


if __name__ == '__main__':
    df1 = to_dataframe(MODEL_DATA_FILE_PATH)
    print('label class ratio:', get_class_ratio(df1))
    print(_configs)
