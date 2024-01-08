import pandas as pd

from cdk8classifier.commons import utils, configs, TRAINING_ROOT_PATH, PRODUCTION_ROOT_PATH
from cdk8classifier.featureengineering import chem_utils


class FingerprintVector:
    def __init__(self, **kwargs):
        self._is_production = kwargs['is_production']
        self._molecule_smiles = kwargs['molecule_smiles']
        self._molecule_y_labels = kwargs['molecule_y_labels']

    def generate_df(self, save_csv=True) -> pd.DataFrame:
        df = generate_bit_df(self._molecule_smiles)
        if not self._is_production:
            df[configs['y_label']] = self._molecule_y_labels

        if save_csv:
            if self._is_production:
                save_to = PRODUCTION_ROOT_PATH
            else:
                save_to = TRAINING_ROOT_PATH
            utils.to_csv(df, "cdk8-bits.csv", save_to=save_to)

        return df


def generate_bit_df(mol_smiles_series) -> pd.DataFrame:
    bit_list = []
    column_list = [str(f'bitvector{x}') for x in range(1024)]

    for _, row in mol_smiles_series.iteritems():
        fp_arr = chem_utils.smiles_to_nparray(row)
        bit_list.append(fp_arr.tolist())

    df = pd.DataFrame(bit_list)
    df.columns = column_list
    return df
