from pathlib import PurePosixPath

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from cdk8classifier import logger
from cdk8classifier.commons import configs, PRODUCTION_ROOT_PATH, PROJECT_ROOT_PATH
from cdk8classifier.commons.exceptions import RDKitSmilesParseError


# count matches between one molecule and multiple substructures and sum them up
def substructure_list_match_count(mol_smiles, substructure_smiles_list) -> int:
    return sum([substructure_match_count(mol_smiles, s) for s in substructure_smiles_list])


# match counts between one molecule and one substructure
def substructure_match_count(mol_smiles, substructure_smiles) -> int:
    mol = Chem.MolFromSmiles(mol_smiles)
    # fragments are SMARTS strings
    substructure = Chem.MolFromSmarts(substructure_smiles)
    matches = mol.GetSubstructMatches(substructure)  # each match is a tuple which is atom indices
    return len(matches)


def smiles_series_to_fingerprints(pd_series) -> list:
    fp_list = []
    for _, row in pd_series.iteritems():
        fp = smiles_to_fingerprint(row)
        fp_list.append(fp)
    return fp_list


def smiles_to_fingerprint(smiles):
    bit_info = {}
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RDKitSmilesParseError(smiles)

    try:
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024, bitInfo=bit_info)
    except Exception as error:
        logger.log_error(msg=f'error with smiles {smiles}, original error: {error}')
        return None


def smiles_to_nparray(smiles) -> np.ndarray:
    fingerprint = smiles_to_fingerprint(smiles)
    array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def sdf_to_smiles(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    smiles_list = []
    for mol in supplier:
        if mol is not None:  # some compounds cannot be loaded.
            smiles_list.append(Chem.MolToSmiles(mol))
    return smiles_list


def smiles_exists_in_file(smiles, file, smiles_name=configs['cdk8']['smiles_name']):
    df = pd.read_csv(file)
    return smiles in df[smiles_name]


# from rdkit.Chem import rdDepictor
# from rdkit.Chem.Draw import rdMolDraw2D
# def mol_to_svg(mol, molSize=(100, 100), kekulize=True):
#     mc = Chem.Mol(mol.ToBinary())
#     if kekulize:
#         try:
#             Chem.Kekulize(mc)
#             # Chem.KekulizeIfPossible(mc) # try
#         except Exception as error:
#             logger.log_error(msg=f'error in mol_to_svg: {error}')
#             mc = Chem.Mol(mol.ToBinary())
#
#     if not mc.GetNumConformers():
#         rdDepictor.Compute2DCoords(mc)
#
#     drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
#     drawer.DrawMolecule(mc)
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText()
#     return svg.replace('svg:', '')


if __name__ == '__main__':
    df = pd.read_csv(PurePosixPath(PROJECT_ROOT_PATH) / 'resources' / 'fragment-smiles.csv')
    target = 'CC(=C)C1CCC2(CO)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC34C)C12'
    count = substructure_list_match_count(target, list(df['Fragment']))
    print(count)

    for f in df['Fragment']:
        c = substructure_match_count(target, f)
        if c > 0:
            print(c, f)

    mol = Chem.MolFromSmiles(target)
    substructure = Chem.MolFromSmarts('C12CCC(C)CC1CCCC2C')
    matches = mol.GetSubstructMatches(substructure)
    print(len(matches), matches)
