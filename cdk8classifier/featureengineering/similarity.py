import numpy as np
import pandas as pd
from rdkit import DataStructs

from cdk8classifier import logger
from cdk8classifier.commons import utils, configs, TRAINING_ROOT_PATH, PRODUCTION_ROOT_PATH
from cdk8classifier.featureengineering import chem_utils


# Generate a dataframe of Tanimoto Similarities of molecules (axis=0) and fragments (axis=1),
# and the last column of data is the accumulated match count of fragments for one molecule.
class Similarity:
    def __init__(self, **kwargs):
        self._is_production = kwargs['is_production']
        self._molecule_smiles = kwargs['molecule_smiles']
        self._molecule_y_labels = kwargs['molecule_y_labels']
        self._substructure_smiles = kwargs['substructure_smiles']
        self._substructure_columns = kwargs['substructure_columns']

    def generate_df(self, save_csv=True) -> pd.DataFrame:
        columns = list(self._substructure_columns)
        columns.append('Match Count')

        df, errors = generate_sim_df(self._molecule_smiles, self._substructure_smiles, columns)
        if not self._is_production:
            df[configs['y_label']] = self._molecule_y_labels

        if errors:
            logger.log_error(msg=f"errors in these fragments and not able to get fingerprint: {errors}")

        if save_csv:
            if self._is_production:
                save_to = PRODUCTION_ROOT_PATH
            else:
                save_to = TRAINING_ROOT_PATH
            utils.to_csv(df, "fragment-similarities.csv", save_to=save_to)

        return df


def generate_sim_df(mol_smiles_series, substructure_smiles_series, columns) -> (pd.DataFrame, list):
    fps_compared_to = chem_utils.smiles_series_to_fingerprints(substructure_smiles_series)
    data = []
    errors = []

    for _, row in mol_smiles_series.iteritems():
        target_fp = chem_utils.smiles_to_fingerprint(row)
        if target_fp:
            sim_rates = DataStructs.BulkTanimotoSimilarity(target_fp, fps_compared_to)
            # round to 9 decimal places
            rounded_rates = list(np.around(np.array(sim_rates), 9))
            count = chem_utils.substructure_list_match_count(row, list(substructure_smiles_series))
            rounded_rates.append(count)
            data.append(rounded_rates)
        else:
            errors.append(row)  # collect a list for debugging

    return pd.DataFrame(columns=columns, data=data), errors
